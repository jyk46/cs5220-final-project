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

struct connection_table {
    connection_table() : rows_(0), cols_(0) {}
    connection_table(const bool *ar, size_t rows, size_t cols) : connected_(rows * cols), rows_(rows), cols_(cols) {
        std::copy(ar, ar + rows * cols, connected_.begin());
    }

    bool is_connected(size_t x, size_t y) const {
        return is_empty() ? true : connected_[y * cols_ + x];
    }

    bool is_empty() const {
        return rows_ == 0 && cols_ == 0;
    }

    std::vector<bool> connected_;
    size_t rows_;
    size_t cols_;
};

enum class padding {
    valid, ///< use valid pixels of input
    same   ///< add zero-padding around input so as to keep image size
};

template<typename Activation = activation::identity, typename Filter = filter_none>
class convolutional_layer : public partial_connected_layer<Activation> {
public:
    typedef partial_connected_layer<Activation> Base;
    CNN_USE_LAYER_MEMBERS;
    using partial_connected_layer<Activation>::scale_factor_;
    using partial_connected_layer<Activation>::out2wi_;
    using partial_connected_layer<Activation>::out2bias_;

    /**
     * constructing convolutional layer
     *
     * @param in_width     [in] input image width
     * @param in_height    [in] input image height
     * @param window_size  [in] window(kernel) size of convolution
     * @param in_channels  [in] input image channels (grayscale=1, rgb=3)
     * @param out_channels [in] output image channels
     * @param padding      [in] rounding strategy
     *                          valid: use valid pixels of input only. output-size = (in-width - window_size + 1) * (in-height - window_size + 1) * out_channels
     *                          same: add zero-padding to keep same width/height. output-size = in-width * in-height * out_channels
     **/
    convolutional_layer(layer_size_t in_width,
                        layer_size_t in_height,
                        layer_size_t window_size,
                        layer_size_t in_channels,
                        layer_size_t out_channels,
                        padding pad_type = padding::valid)
    : Base(in_width * in_height * in_channels,
           out_size(in_width, in_height, window_size, pad_type) * out_channels, 
           sqr(window_size) * in_channels * out_channels, out_channels,
           in_width, in_height, in_channels,
           out_length(in_width, window_size, pad_type),
           out_length(in_height, window_size, pad_type),
           out_channels),
      in_(in_width, in_height, in_channels),
      out_(out_length(in_width, window_size, pad_type), out_length(in_height, window_size, pad_type), out_channels),
      weight_(window_size, window_size, in_channels*out_channels),
      window_size_(window_size)
    {
        init_connection(connection_table(), pad_type);
    }

    /**
     * constructing convolutional layer
     *
     * @param in_width         [in] input image width
     * @param in_height        [in] input image height
     * @param window_size      [in] window(kernel) size of convolution
     * @param in_channels      [in] input image channels (grayscale=1, rgb=3)
     * @param out_channels     [in] output image channels
     * @param connection_table [in] definition of connections between in-channels and out-channels
     * @param pad_type         [in] rounding strategy 
     *                               valid: use valid pixels of input only. output-size = (in-width - window_size + 1) * (in-height - window_size + 1) * out_channels
     *                               same: add zero-padding to keep same width/height. output-size = in-width * in-height * out_channels
     **/
    convolutional_layer(layer_size_t in_width,
                        layer_size_t in_height,
                        layer_size_t window_size,
                        layer_size_t in_channels,
                        layer_size_t out_channels,
                        const connection_table& connection_table,
                        padding pad_type = padding::valid)
        : Base(in_width * in_height * in_channels,
               out_size(in_width, in_height, window_size, pad_type) * out_channels,
               sqr(window_size) * in_channels * out_channels, out_channels,
               in_width, in_height, in_channels,
               out_length(in_width, window_size, pad_type),
               out_length(in_height, window_size, pad_type),
               out_channels),
          in_(in_width, in_height, in_channels),
          out_(out_length(in_width, window_size, pad_type), out_length(in_height, window_size, pad_type), out_channels),
          weight_(window_size, window_size, in_channels*out_channels),
          connection_(connection_table),
          window_size_(window_size)
    {
        init_connection(connection_table, pad_type);
        this->remap();
    }

    const vec_t& forward_propagation(const vec_t& in, size_t index) override {

        // Broadcast scale factor to vector register
        vec scale_vec = vec_set1(scale_factor_);

        // Pointers to per-thread aligned input/output buffers to allow
        // vector loads/stores.
        float_t* in_buf  = aligned_in_[index];
        float_t* out_buf = aligned_out_[index];

        // Copy data from the unaligned input buffer to the aligned input
        // buffer. The aligned output buffer is filled in as computation
        // progresses so we do not need to copy anything to the output
        // buffer.
        pack_padded_data(
            in_width_, in_height_, in_channels_,
            in_width_padded_, in_buf, &in[0]
        );

        // Outer loop iterates across the rows in the output buffer, where
        // each channel has dimensions of out_width_ x out_height_, and
        // there are out_channels_ channels. Inner loop vectorizes
        // computation across the columns within a row. All rows are
        // assumed to be padded to the vector length.

        for (int i = 0; i < out_height_ * out_channels_; i++ ) {
          for (int j = 0; j < out_width_vecs_; j++ ) {

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
              // the input values.

              int aligned_in_i = calculate_aligned_i(
                  in_width_, in_width_padded_, connection.second);

              float_t* in_addr = in_buf + aligned_in_i;
              vec      in_vec  = vec_load(in_addr);

              // Multiply and accumulate the partial products across
              // vectorized frontier.
              out_vec = vec_add(out_vec, vec_mul(weight_vec, in_vec));

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

    image<> weight_to_image() const {
        image<> img;
        const layer_size_t border_width = 1;
        const auto pitch = window_size_ + border_width;
        const auto width = out_.depth_ * pitch + border_width;
        const auto height = in_.depth_ * pitch + border_width;
        const image<>::intensity_t bg_color = 255;

        img.resize(width, height);
        img.fill(bg_color);

        auto minmax = std::minmax_element(this->W_.begin(), this->W_.end());

        for (layer_size_t r = 0; r < in_.depth_; ++r) {
            for (layer_size_t c = 0; c < out_.depth_; ++c) {
                if (!connection_.is_connected(c, r)) continue;

                const auto top = r * pitch + border_width;
                const auto left = c * pitch + border_width;

                for (layer_size_t y = 0; y < window_size_; ++y) {
                    for (layer_size_t x = 0; x < window_size_; ++x) {
                        const float_t w = W_[weight_.get_index(x, y, c * in_.depth_ + r)];

                        img.at(left + x, top + y)
                            = static_cast<image<>::intensity_t>(rescale(w, *minmax.first, *minmax.second, 0, 255));
                    }
                }
            }
        }
        return img;
    }

    index3d<layer_size_t> in_shape() const override { return in_; }
    index3d<layer_size_t> out_shape() const override { return out_; }
    std::string layer_type() const override { return "conv"; }

private:
    layer_size_t out_length(layer_size_t in_length, layer_size_t window_size, padding pad_type) const {
        return pad_type == padding::same ? in_length : (in_length - window_size + 1);
    }

    layer_size_t out_size(layer_size_t in_width, layer_size_t in_height, layer_size_t window_size, padding pad_type) const {
        return out_length(in_width, window_size, pad_type) * out_length(in_height, window_size, pad_type);
    }

    void init_connection(const connection_table& table, padding pad_type) {
        layer_size_t pad = (pad_type == padding::valid) ? 0 : window_size_ / 2;

        for (layer_size_t inc = 0; inc < in_.depth_; ++inc) {
            for (layer_size_t outc = 0; outc < out_.depth_; ++outc) {
                if (!table.is_connected(outc, inc)) {
                    continue;
                }

                for (layer_size_t y = 0; y < out_.height_; ++y)
                    for (layer_size_t x = 0; x < out_.width_; ++x)
                        connect_kernel(inc, outc, x, y, pad);
            }
        }

        for (layer_size_t outc = 0; outc < out_.depth_; ++outc)
            for (layer_size_t y = 0; y < out_.height_; ++y)
                for (layer_size_t x = 0; x < out_.width_; ++x)
                    this->connect_bias(outc, out_.get_index(x, y, outc));
    }

    void connect_kernel(layer_size_t inc, layer_size_t outc, layer_size_t x, layer_size_t y, layer_size_t pad) {

        for (layer_size_t dy = 0; dy < window_size_; ++dy) {
            if (y + dy < pad) continue;
            if (y + dy - pad >= in_.height_) continue;

            for (layer_size_t dx = 0; dx < window_size_; ++dx) {
                if (x + dx < pad) continue;
                if (x + dx - pad >= in_.width_) continue;

                this->connect_weight(
                    in_.get_index(x + dx - pad, y + dy - pad, inc), 
                    out_.get_index(x, y, outc), 
                    weight_.get_index(dx, dy, outc * in_.depth_ + inc));
            }
        }
    }

    index3d<layer_size_t> in_;
    index3d<layer_size_t> out_;
    index3d<layer_size_t> weight_;
    connection_table connection_;
    size_t window_size_;
};

} // namespace tiny_cnn
