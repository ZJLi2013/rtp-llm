
#include "gtest/gtest.h"

#define private public
#include "maga_transformer/cpp/system_prompt/SystemPrompt.h"
#include "maga_transformer/cpp/system_prompt/SystemPromptConstructor.h"
#include "src/fastertransformer/devices/testing/TestBase.h"
#include "src/fastertransformer/core/Tensor.h"

#include <memory>
#include <thread>
#include <chrono>

using namespace std;

namespace rtp_llm {

class SystemPromptTest : public DeviceTestBase {
protected:

protected:
};

TEST_F(SystemPromptTest, testConstruct) {
}

}
