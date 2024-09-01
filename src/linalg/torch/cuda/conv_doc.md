
卷积包括这么几个内容：
1. kernel 大小
2. padding
3. stride 每一次移动多少步（kernel 中心的移动）
4. dilate 卷积元素的间隔

除了 kernel 大小之外，还有多通道卷积：
1. 输入通道
2. 输出通道

注意多通道卷积的逻辑：假设输入通道是 3，输出通道是 8。这就意味着，一共有 8 个 k * k 的三通道卷积核，把某个位置 3 * k * k 的信息进行综合后。输出到输出通道中的某一个值中，重复八次

如果要更好地理解，可以看卷积可视化器：https://ezyang.github.io/convolution-visualizer/


CPP 伪代码逻辑:
```c++
class Tensor {
public:
    Tensor(int w, int h) {}

    float& operator() (int i, int j, int k) {

    }

    const float& operator() (int i, int j, int k) const {

    }
    
    int get_w() const {}
    int get_h() const {}
    int get_kernel_size() const {}
    int get_padding()     const {}
    int get_dilate()      const {}
    int get_stride()      const {}
    int get_channel()     const {}
};


void padding_input(Tensor& tensor, int padding) {
    // 具体是分配一个更大的 tensor，后将 tensor 的数据复制到对应的起始位置
    // 需要逐行复制，dilate 后原本连续的存储变得不连续了
}

int get_steps(int w, const Tensor& kernel) {
    // output shape 的计算：某一边： (w + padding * 2 - (dilation * kernel_size - kernel_size + 1) / stride)
}

// 这里仅研究单通道输出的卷积
Tensor convolution(const Tensor& input, const Tensor& kernel) {
    int step_w = get_steps(input.get_w(), kernel),
        step_h = get_steps(input.get_h(), kernel);
    Tensor output(step_w, step_h);
    for (int i_y = 0; i_y < step_h; i_y ++) {
        int y_base = i_y * kernel.get_stride();
        for (int i_x = 0; i_x < step_w; i_x ++) {
            int x_base = i_x * kernel.get_stride();
            float val_sum = 0;
            for (int channel = 0; channel < input.get_channel(); channel++) {
                // kernel 可以放 shared memory 中？
                for (int kw = 0; kw < kernel.get_kernel_size(); kw++) {
                    int kw_base = kw * kernel.get_dilate();
                    for (int kh = 0; kh < kernel.get_kernel_size(); kh++) {
                        int kh_base = kh * kernel.get_dilate();
                        val_sum += input(x_base + kw_base, y_base + kh_base, channel) * kernel(kw, kh, channel);
                    }
                }
            }
            output(i_x, i_y, 0) = val_sum;
        }
    }
    return output;
}
```

