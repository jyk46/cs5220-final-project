/*
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY 
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND 
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
#include "util.h"
#include "partial_connected_layer.h"
#include "image.h"
#include "activation_function.h"

namespace tiny_cnn {


template<typename Activation = activation::identity>
class average_pooling_layer : public partial_connected_layer<Activation> {
public:
    typedef partial_connected_layer<Activation> Base;
    CNN_USE_LAYER_MEMBERS;
    using partial_connected_layer<Activation>::scale_factor_;
    using partial_connected_layer<Activation>::out2wi_;
    using partial_connected_layer<Activation>::out2bias_;

    average_pooling_layer(layer_size_t in_width, layer_size_t in_height, layer_size_t in_channels, layer_size_t pooling_size)
    : Base(in_width * in_height * in_channels,
           in_width * in_height * in_channels / sqr(pooling_size),
           in_channels, in_channels,
           in_width, in_height, in_channels,
           in_width/pooling_size, in_height/pooling_size, in_channels,
           pooling_size, 1.0 / sqr(pooling_size)),
      in_(in_width, in_height, in_channels), 
      out_(in_width/pooling_size, in_height/pooling_size, in_channels)
    {
        if ((in_width % pooling_size) || (in_height % pooling_size))
            pooling_size_mismatch(in_width, in_height, pooling_size);

        init_connection(pooling_size);

        // Allocate special aligned input buffer with more padding for
        // strided loads.

        int strided_vlen_nelem       = pooling_size * CNN_VLEN_NELEM;
            strided_in_width_vecs_   = (in_width_ + strided_vlen_nelem - 1) / strided_vlen_nelem;
            strided_in_width_padded_ = strided_in_width_vecs_ * strided_vlen_nelem;
        int strided_in_nbytes        = strided_in_width_padded_ * in_height_ * in_channels_ * sizeof(float_t);

        for (int i = 0; i < CNN_TASK_SIZE; i++)
          aligned_strided_in_[i]  = (float_t*)_mm_malloc(strided_in_nbytes, CNN_VLEN_NBYTES);
    }

    // Vectorized forward propagation
    const vec_t& forward_propagation(const vec_t& in, size_t index) override {

        // Broadcast scale factor to vector register
        vec scale_vec = vec_set1(scale_factor_);

        // Set stride for input load
        veci stride_vec = vec_set(
          #ifdef CNN_USE_AVX512
          window_width_ * 7,
          window_width_ * 6,
          window_width_ * 5,
          window_width_ * 4,
          #endif
          window_width_ * 3,
          window_width_ * 2,
          window_width_ * 1,
          window_width_ * 0
        );

        // Pointers to per-thread aligned input/output buffers to allow
        // vector loads/stores.
        float_t* in_buf  = aligned_strided_in_[index];
        float_t* out_buf = aligned_out_[index];

        // Copy data from the unaligned input buffer to the aligned input
        // buffer. The aligned output buffer is filled in as computation
        // progresses so we do not need to copy anything to the output
        // buffer.
        pack_padded_data(
            in_width_, in_height_, in_channels_,
            strided_in_width_padded_, in_buf, &in[0]
        );

        // Outer loop iterates across the rows in the output buffer, where
        // each channel has dimensions of out_width_ x out_height_, and
        // there are out_channels_ channels. Inner loop vectorizes
        // computation across the columns within a row. All rows are
        // assumed to be padded to the vector length.

        for (int i = 0; i < out_height_ * out_channels_; i++) {
          for (int j = 0; j < out_width_vecs_; j++) {

            // Calculate both unaligned/aligned offsets to the output
            // buffer. The former is used to index into the connections
            // array whereas the latter is used to address into the
            // aligned output buffer.
            int out_i         = (i * out_width_) + (j * CNN_VLEN_NELEM);
            int aligned_out_i = (i * out_width_padded_) + (j * CNN_VLEN_NELEM);

            // Initialize output partial product register
            vec out_vec = vec_setzero();

            // Load the connections between the weights and inputs for
            // this layer. first -> weight, second -> input. Multiple
            // elements in the current row multiply the shared weights
            // with the corresponding inputs and accumulate the partial
            // products across all connections (e.g., 5x5x1 filter).

            for (auto connection : out2wi_[out_i]) {

              // Load and broadcast shared weights for current connection
              float_t weight     = W_[connection.first];
              vec     weight_vec = vec_set1(weight);

              // Since the connection object returns the input index into
              // the unaligned input buffer, we need to convert this into
              // the index into the aligned input buffer before loading
              // the input values. We do a gather with a stride
              // corresponding to the pooling size for this layer (always
              // load doubles = 8B).

              int aligned_in_i = calculate_aligned_i(
                  in_width_, strided_in_width_padded_, connection.second);

              float_t* in_addr = in_buf + aligned_in_i;

              // Annoyingly, AVX-512 version of the gather flips the
              // vindex and base_addr arguments around.
              #ifdef CNN_USE_AVX512
              vec in_vec = vec_gather(stride_vec, in_addr, 8);
              #else
              vec in_vec = vec_gather(in_addr, stride_vec, 8);
              #endif

              // Multiply and accumulate the partial products across
              // vectorized frontier.
              out_vec = vec_fmadd(weight_vec, in_vec, out_vec);

            }

            // Apply constant scale factor
            out_vec = vec_mul(out_vec, scale_vec);

            // Store the final outputs to aligned output buffer
            float_t* out_addr = out_buf + aligned_out_i;
            vec_store(out_addr, out_vec);

          }
        }

        // Apply bias and activation function
        for (int i = 0; i < out_height_ * out_channels_; i++ ) {
          for (int j = 0; j < out_width_; j++ ) {
            int out_i         = (i * out_width_) + j;
            int aligned_out_i = (i * out_width_padded_) + j;

            out_buf[aligned_out_i] += b_[out2bias_[out_i]];

            output_[index][out_i] = h_.f(out_buf, aligned_out_i);
          }
        }

        return next_ ? next_->forward_propagation(output_[index], index) : output_[index]; // 15.6%
    }

    image<> output_to_image(size_t worker_index = 0) const override {
        return vec2image<unsigned char>(output_[worker_index], out_);
    }

    index3d<layer_size_t> in_shape() const override { return in_; }
    index3d<layer_size_t> out_shape() const override { return out_; }
    std::string layer_type() const override { return "ave-pool"; }

private:
    void init_connection(layer_size_t pooling_size) {
        for (layer_size_t c = 0; c < in_.depth_; ++c)
            for (layer_size_t y = 0; y < in_.height_; y += pooling_size)
                for (layer_size_t x = 0; x < in_.width_; x += pooling_size)
                    connect_kernel(pooling_size, x, y, c);


        for (layer_size_t c = 0; c < in_.depth_; ++c)
            for (layer_size_t y = 0; y < out_.height_; ++y)
                for (layer_size_t x = 0; x < out_.width_; ++x)
                    this->connect_bias(c, out_.get_index(x, y, c));
    }

    void connect_kernel(layer_size_t pooling_size, layer_size_t x, layer_size_t y, layer_size_t inc) {
        for (layer_size_t dy = 0; dy < pooling_size; ++dy)
            for (layer_size_t dx = 0; dx < pooling_size; ++dx)
                this->connect_weight(
                    in_.get_index(x + dx, y + dy, inc), 
                    out_.get_index(x / pooling_size, y / pooling_size, inc),
                    inc);
    }

    index3d<layer_size_t> in_;
    index3d<layer_size_t> out_;

    // Special aligned buffer

    size_t strided_in_width_vecs_;
    size_t strided_in_width_padded_;

    float_t* aligned_strided_in_[CNN_TASK_SIZE];
};

} // namespace tiny_cnn
