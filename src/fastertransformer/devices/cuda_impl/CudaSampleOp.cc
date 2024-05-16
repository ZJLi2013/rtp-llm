#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/CommonDefines.h"
#include "src/fastertransformer/kernels/sampling_topk_kernels.h"
#include "src/fastertransformer/kernels/sampling_topp_kernels.h"
#include "src/fastertransformer/kernels/sampling_penalty_kernels.h"
#include "src/fastertransformer/cuda/memory_utils.h"

using namespace std;

namespace fastertransformer {

using SamplerT = float;

// batch sampling explained:
// topk = [4, 0, 4]. topp = [0.0, 0.5, 0.5]
// then topk_decode handles [4, x, 4 + 0.5]
//      topp_decode handles [x, 0.5, x]
// where "x" are skipped.
// topk should has higher proirity than topp.

void CudaDevice::sampleGreedy(const GreedyParams& params) {
    const auto& logits = params.logits;
    const auto batch_size = logits.shape()[0];
    const auto vocab_size_padded = logits.shape()[1];
    const auto step = params.step;
    RUNTIME_ASSERT_OP_ARG(batch_size == params.token_ids.shape()[0],
                          "logits.shape[0] should equal to token_ids.shape[0], but %d vs %d",
                          batch_size, params.token_ids.shape()[0]);
    RUNTIME_ASSERT_OP_ARG((step == params.token_ids.shape()[1] - 1),
                          "step should equal to token_ids.shape[1] - 1, but %d vs %d",
                          step, params.token_ids.shape()[1] - 1);
    auto transposed_tokens = transpose({params.token_ids});

    // 1. confirm buffer sizes
    auto& top_k = params.top_k;
    auto& top_p = params.top_p;
    auto& temperature = params.temperature;
    auto& random_seed = params.random_seed;
    assert(top_k.size() == batch_size);
    assert(top_p.size() == batch_size);
    assert(temperature.size() == batch_size);

    auto default_top_k = 1;
    auto max_top_k = *max_element(top_k.data<int32_t>(), top_k.dataWithOffset<int32_t>(top_k.size()));
    if (max_top_k == 0) {
        // for safety. TopKSamplingLayer handles a case of top_k=0 and top_p=0 as
        // a greedy decode, i.e. top_k=1, although such case has max_top_k=0.
        max_top_k = 1;
    }
    auto max_top_p = *max_element(top_p.data<SamplerT>(), top_p.dataWithOffset<SamplerT>(top_p.size()));
    FT_LOG_DEBUG("max_top_k: %d, max_top_p: %f", max_top_k, max_top_p);

    size_t topk_ws_size;
    size_t topp_ws_size;
    size_t cub_temp_storage_size; // useless variable

    // these two kernel calls are only for querying workspace size,
    // all the args are ignored.
    invokeTopKSampling<SamplerT>(nullptr,
                                topk_ws_size,
                                nullptr,
                                nullptr,
                                nullptr,
                                nullptr,
                                nullptr,
                                nullptr,
                                nullptr,
                                nullptr,
                                nullptr,
                                max_top_k,
                                1.0f,
                                vocab_size_padded,
                                nullptr,
                                stream_,
                                batch_size,
                                nullptr);

    invokeTopPSampling<SamplerT>(nullptr,  // workspace
                                topp_ws_size,
                                cub_temp_storage_size,
                                nullptr,
                                nullptr,
                                nullptr,
                                nullptr,
                                nullptr,
                                nullptr,
                                nullptr,
                                nullptr,
                                nullptr,
                                nullptr,
                                batch_size,
                                vocab_size_padded,
                                nullptr,
                                max_top_p,
                                stream_,
                                &device_prop_,
                                nullptr);

    FT_LOG_DEBUG("topk_ws_size: %d, topp_ws_size: %d", topk_ws_size, topp_ws_size);

    auto default_top_p = 0.0f;
    // 2. allocate buffers

    // see BaseSamplingLayer<T>::allocateBuffer ------------------
    auto curandstate_buf = allocateBuffer({batch_size * sizeof(curandState_t)});
    auto skip_top_k_decode_buf = allocateBuffer({DataType::TYPE_BOOL, {batch_size}});
    auto skip_top_p_decode_buf = allocateBuffer({DataType::TYPE_BOOL, {batch_size}});

    auto initial_top_p_buf = allocateBuffer({DataType::TYPE_FP32, {batch_size}});
    auto topp_id_vals_buf = allocateBuffer({DataType::TYPE_INT32, {batch_size * vocab_size_padded}});
    auto topp_offset_buf = allocateBuffer({DataType::TYPE_INT32, {batch_size + 1}});
    auto begin_topp_offset_buf = allocateBuffer({DataType::TYPE_INT32, {batch_size + 1}});

    // TopKSamplingLayer<T>::allocateBuffer
    auto top_k_workspace = allocateBuffer({topk_ws_size});
    auto top_p_workspace = allocateBuffer({topp_ws_size});
    auto runtime_top_k_buf = allocateBuffer({DataType::TYPE_UINT32, {batch_size}});
    copy({*runtime_top_k_buf, top_k});
    auto runtime_top_p_buf = allocateBuffer({DataType::TYPE_FP32, {batch_size}});
    copy({*runtime_top_p_buf, top_p});

    // 3. prepare kernel inputs

    // 3.1. base sampling layer setup
    // TODO: this curandstate buf might should be moved to cuda device state,
    //       so that do not need to be reinitialized on every request.
    if (random_seed) {
        auto& seeds = random_seed.value().get();
        if (seeds.size() == 1) {
            invokeCurandInitialize(
                (curandState_t *)curandstate_buf->data(), batch_size,
                seeds.data<uint64_t>()[0], stream_);
        } else {
            auto random_seeds_buf = allocateBuffer({DataType::TYPE_UINT64, {batch_size}});
            assert(seeds.size() == batch_size);
            copy({*random_seeds_buf, seeds});
            invokeCurandBatchInitialize(
                (curandState_t *)curandstate_buf->data(), batch_size,
                (unsigned long long *)random_seeds_buf->data(), stream_);
        }
    } else {
        // Initialize curand states using the default seed 0.
        invokeCurandInitialize((curandState_t *)curandstate_buf->data(), batch_size, 0, stream_);
    }

    // 3.2. topk setup
    invokeSetupTopKRuntimeArgs(batch_size,
                               default_top_k,
                               runtime_top_k_buf->data<uint>(),
                               batch_size,
                               default_top_p,
                               runtime_top_p_buf->data<float>(),
                               batch_size,
                               skip_top_k_decode_buf->data<bool>(),
                               stream_);

    // 3.3 top_p setup
    invokeSetupTopPRuntimeArgs(batch_size,
                               default_top_k,
                               runtime_top_k_buf->data<uint>(),
                               batch_size,
                               default_top_p,
                               runtime_top_p_buf->data<float>(),
                               batch_size,
                               skip_top_p_decode_buf->data<bool>(),
                               initial_top_p_buf->data<float>(),
                               nullptr, // top_p_decay_buf,
                               nullptr,
                               nullptr, // top_p_min_buf,
                               nullptr,
                               nullptr, // top_p_reset_ids_buf,
                               nullptr,
                               stream_);
    sync_check_cuda_error();

    // 4. kernel call
    auto cum_log_probs = params.cum_log_probs.has_value() ?
                         params.cum_log_probs.value().get().data<float>() : nullptr;
    auto output_log_probs = params.output_log_probs.has_value() ?
                            params.output_log_probs.value().get().data<float>() : nullptr;

    // base sampling layer forward
    if (std::any_of(temperature.data<float>(),
                    temperature.data<float>() + batch_size,
                    [&](auto t) { return t != 1.0f; }))
    {
        BufferPtr temperature_buf = allocateBuffer({DataType::TYPE_FP32, {batch_size}});
        copy({*temperature_buf, temperature});
        invokeBatchApplyTemperaturePenalty(
            logits.data<float>(),
            (float *)nullptr, // embedding_bias
            temperature_buf->data<float>(),
            batch_size,
            vocab_size_padded,
            vocab_size_padded,
            stream_);
    }

    const auto decoder_batch_size = params.sequence_lengths.shape()[0];
    if (decoder_batch_size) {
        auto sequence_lengths = clone({params.sequence_lengths});
        auto input_lengths = clone({params.input_lengths});

        if (step > 1 && params.repetition_penalty && decoder_batch_size) {
            auto& repetition_penalty = params.repetition_penalty->get();
            if (std::any_of(repetition_penalty.data<float>(),
                            repetition_penalty.data<float>() + batch_size,
                            [&](auto t) { return t == 1.0f; }))
            {
                const auto repetition_penalty_type = RepetitionPenaltyType::Multiplicative;
                auto repetition_penalty_buf = allocateBuffer({DataType::TYPE_FP32, {batch_size}});
                auto penalty_logits = allocateBuffer({DataType::TYPE_FP32, {batch_size * 64 * 1024}});
                copy({*repetition_penalty_buf, repetition_penalty});
                invokeBatchApplyRepetitionPenalty(
                    logits.data<float>(),
                    penalty_logits->data<float>(),
                    repetition_penalty_buf->data<float>(),
                    transposed_tokens->data<int32_t>(),
                    batch_size,
                    batch_size, // local_batch_size
                    vocab_size_padded,
                    sequence_lengths->data<int32_t>(),
                    step - 1, // max_input_length
                    step - 1, // step
                    repetition_penalty_type,
                    stream_);
                // NOTE: here step uses step - 1 is to use same input_lengths as passed into ModelInputs.
            }
        }

        if (params.min_lengths && params.eos_ids) {
            auto min_lengths_buf = clone({params.min_lengths.value().get()});
            invokeMinLengthPenaltyNew(
                logits.data<float>(),
                min_lengths_buf->data<int32_t>(),
                params.eos_ids.value().get().data<int32_t>(),
                sequence_lengths->data<int32_t>(),
                input_lengths->data<int32_t>(),
                decoder_batch_size,
                vocab_size_padded,
                stream_);
        }
    }

    // TODO: test repetition_penalty
    // TODO: test min_new_tokens

    // run top_k
    invokeBatchTopKSampling(
        top_k_workspace->data(),
        topk_ws_size,
        logits.data<float>(),
        transposed_tokens->dataWithOffset<int32_t>(step * batch_size),
        nullptr, // sequence_length
        nullptr, // finished
        cum_log_probs,
        output_log_probs,
        nullptr, // output_index_logits
        nullptr, // token_id_for_index_prob,
        (curandState_t *)curandstate_buf->data(),
        max_top_k,  // useless because runtime_top_k_buf_ is never nullptr. Keep for legacy.
        (int32_t*)runtime_top_k_buf->data<uint32_t>(),
        1.0f,  // useless because runtime_top_p_buf_ is never nullptr. Keep for legacy.
        runtime_top_p_buf->data<float>(),
        vocab_size_padded,
        nullptr, // end_id
        stream_,
        batch_size,
        skip_top_k_decode_buf->data<bool>());

    // run top_p
    invokeTopPInitialize(
        topp_id_vals_buf->data<int32_t>(),
        topp_offset_buf->data<int32_t>(),
        begin_topp_offset_buf->data<int32_t>(),
        batch_size,
        vocab_size_padded,
        stream_);

    invokeAddBiasSoftMax(
        logits.data<SamplerT>(),
        (SamplerT *)nullptr, // bias
        nullptr, // end_id
        nullptr, // finished
        batch_size,
        vocab_size_padded,
        vocab_size_padded,
        stream_);

    invokeBatchTopPSampling(
        top_p_workspace->data(),
        topp_ws_size,
        cub_temp_storage_size,
        transposed_tokens->dataWithOffset<int32_t>(step * batch_size),
        nullptr, // sequence_length
        nullptr, // finished
        cum_log_probs,
        output_log_probs,
        logits.data<float>(),
        topp_id_vals_buf->data<int32_t>(),
        topp_offset_buf->data<int32_t>(),
        begin_topp_offset_buf->data<int32_t>(),
        (curandState_t *)curandstate_buf->data(),
        batch_size,
        vocab_size_padded,
        nullptr, // end_id
        max_top_p,
        runtime_top_p_buf->data<float>(),
        stream_,
        &device_prop_,
        skip_top_p_decode_buf->data<bool>());

    auto output_tokens = transpose({*transposed_tokens});
    copy({params.token_ids, *output_tokens});
    sync_check_cuda_error();
}

} // namespace fastertransformer
