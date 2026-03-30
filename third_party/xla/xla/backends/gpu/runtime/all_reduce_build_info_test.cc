/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/all_reduce.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/all_reduce_kernel.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {  // namespace

using ::stream_executor::gpu::AllReduceStrategy;
using ::testing::HasSubstr;

struct IsSupportedTestParams {
  bool is_collective_kernel_enabled;
  bool is_multimem_enabled;
  PrimitiveType element_type;
  std::vector<int64_t> shape;
  HloOpcode hlo_opcode;
  std::vector<int32_t> replica_groups;
  // Expected status.
  absl::StatusOr<AllReduceInfo> expected;

  static std::vector<IsSupportedTestParams> Generate() {
    return {
        // Success cases
        {/*is_collective_kernel_enabled=*/true,
         /*is_multimem_enabled=*/false, S32,
         /*shape=*/{/*dim0=*/1024, /*dim1=*/8}, HloOpcode::kMaximum,
         /*replica_groups=*/{/*rank0=*/0, /*rank1=*/1},
         AllReduceInfo{ReductionKind::MAX, /*num_devices=*/2,
                       /*num_elements=*/8192, PrimitiveType::S32,
                       AllReduceStrategy::kOneShot}},
        {/*is_collective_kernel_enabled=*/true,
         /*is_multimem_enabled=*/false, F32,
         /*shape=*/{/*dim0=*/128 * 1024}, HloOpcode::kAdd,
         /*replica_groups=*/{/*rank0=*/0, /*rank1=*/1},
         AllReduceInfo{ReductionKind::SUM, /*num_devices=*/2,
                       /*num_elements=*/131072, PrimitiveType::F32,
                       AllReduceStrategy::kTwoShot}},
        {/*is_collective_kernel_enabled=*/true,
         /*is_multimem_enabled=*/true, F32,
         /*shape=*/{/*dim0=*/1024}, HloOpcode::kAdd,
         /*replica_groups=*/{/*rank0=*/0, /*rank1=*/1},
         AllReduceInfo{ReductionKind::SUM, /*num_devices=*/2,
                       /*num_elements=*/1024, PrimitiveType::F32,
                       AllReduceStrategy::kMultimem}},

        // Failure cases
        {/*is_collective_kernel_enabled=*/false,
         /*is_multimem_enabled=*/false, F32,
         /*shape=*/{/*dim0=*/1024}, HloOpcode::kAdd,
         /*replica_groups=*/{/*rank0=*/0, /*rank1=*/1},
         absl::UnimplementedError("Collective kernel is not enabled.")},
        {/*is_collective_kernel_enabled=*/true,
         /*is_multimem_enabled=*/false, F32,
         /*shape=*/{/*dim0=*/1024}, HloOpcode::kAdd,
         /*replica_groups=*/{0, 1, 2},
         absl::UnimplementedError("only supported for power of 2")},
        {/*is_collective_kernel_enabled=*/true,
         /*is_multimem_enabled=*/false,
         F32,
         /*shape=*/{/*dim0=*/1024},
         HloOpcode::kAdd,
         /*replica_groups=*/
         {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
         absl::UnimplementedError("does not support more than 8 ranks")},
        {/*is_collective_kernel_enabled=*/true,
         /*is_multimem_enabled=*/false, PRED,
         /*shape=*/{/*dim0=*/1024}, HloOpcode::kAdd,
         /*replica_groups=*/{0, 1},
         absl::UnimplementedError("combination is not supported")},
        {/*is_collective_kernel_enabled=*/true,
         /*is_multimem_enabled=*/false, F32,
         /*shape=*/{/*dim0=*/2 * 1024 * 1024}, HloOpcode::kAdd,
         /*replica_groups=*/{0, 1},
         absl::UnimplementedError("only supported for small inputs")},
        {/*is_collective_kernel_enabled=*/true,
         /*is_multimem_enabled=*/false, F32,
         /*shape=*/{/*dim0=*/1024}, HloOpcode::kAdd,
         /*replica_groups=*/{},
         absl::UnimplementedError(
             "Replica groups must be explicitly provided")},
    };
  }

  [[maybe_unused]] friend void PrintTo(const IsSupportedTestParams& params,
                                       std::ostream* os) {
    *os << "{ .element_type=" << absl::StrFormat("%v", params.element_type)
        << ", .shape=" << absl::StrJoin(params.shape, ",")
        << ", .opcode=" << absl::StrFormat("%v", params.hlo_opcode)
        << ", .coll_enabled=" << params.is_collective_kernel_enabled
        << ", .multimem_enabled=" << params.is_multimem_enabled
        << ", .replica_groups=" << absl::StrJoin(params.replica_groups, ",")
        << " }";
  }
};

class BuildAllReduceInfoTest
    : public HloHardwareIndependentTestBase,
      public ::testing::WithParamInterface<IsSupportedTestParams> {};

INSTANTIATE_TEST_SUITE_P(
    BuildAllReduceInfoTest, BuildAllReduceInfoTest,
    ::testing::ConvertGenerator(
        ::testing::ValuesIn(IsSupportedTestParams::Generate()),
        [](const IsSupportedTestParams& params) { return params; }),
    [](const ::testing::TestParamInfo<IsSupportedTestParams>& info) {
      return absl::StrFormat(
          "%s_%s_%s_%s_%s_R%d",
          primitive_util::LowercasePrimitiveTypeName(info.param.element_type),
          absl::StrJoin(info.param.shape, "_"),
          HloOpcodeString(info.param.hlo_opcode),
          info.param.is_collective_kernel_enabled ? "collectivekernelenabled"
                                                  : "nocollectivekernelenabled",
          info.param.is_multimem_enabled ? "multimem" : "nomultimem",
          info.param.replica_groups.size());
    });

TEST_P(BuildAllReduceInfoTest, BuildAllReduceInfo) {
  const IsSupportedTestParams& params = GetParam();
  constexpr absl::string_view kModuleStr = R"(
  HloModule test
   apply_op {
     x = %1$s[] parameter(0)
     y = %1$s[] parameter(1)
     ROOT apply_op = %1$s[] %3$s(x, y)
   }
   ENTRY test_computation {
     param_0 = %1$s[%2$s] parameter(0)
     ROOT all-reduce = %1$s[%2$s] all-reduce(param_0), to_apply=apply_op,
         replica_groups={%4$s}
   }
  )";
  se::DeviceDescription device_info = TestGpuDeviceInfo::H100SXMDeviceInfo();
  std::string replica_groups_str =
      params.replica_groups.empty()
          ? ""
          : absl::StrFormat("{%s}", absl::StrJoin(params.replica_groups, ","));
  const std::string module_str = absl::StrFormat(
      kModuleStr,
      primitive_util::LowercasePrimitiveTypeName(params.element_type),
      absl::StrJoin(params.shape, ","), HloOpcodeString(params.hlo_opcode),
      replica_groups_str);
  SCOPED_TRACE(::testing::Message() << "module_str: " << module_str);
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(
          module_str,
          params.replica_groups.empty() ? 1 : params.replica_groups.size()));
  const HloInstruction* const hlo_instr =
      HloHardwareIndependentTestBase::FindInstruction(module.get(),
                                                      HloOpcode::kAllReduce);
  ASSERT_NE(hlo_instr, nullptr);
  const HloAllReduceInstruction* instr =
      Cast<HloAllReduceInstruction>(hlo_instr);
  auto result =
      BuildAllReduceInfo(params.is_collective_kernel_enabled,
                         params.is_multimem_enabled, device_info, instr);
  if (params.expected.ok()) {
    ASSERT_OK(result.status());
    EXPECT_EQ(result->reduction_kind, params.expected->reduction_kind);
    EXPECT_EQ(result->num_devices, params.expected->num_devices);
    EXPECT_EQ(result->num_elements, params.expected->num_elements);
    EXPECT_EQ(result->element_type, params.expected->element_type);
    EXPECT_EQ(result->all_reduce_strategy,
              params.expected->all_reduce_strategy);
  } else {
    EXPECT_THAT(
        result.status(),
        absl_testing::StatusIs(params.expected.status().code(),
                               HasSubstr(params.expected.status().message())));
  }
}
}  // namespace
}  // namespace xla::gpu
