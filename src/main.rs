use cuda::runtime as cuda;

fn main() {
    // 获取当前系统中的第一个NVIDIA GPU
    let device = cuda::Device::get_device(0).unwrap();
    println!("GPU: {}", device.name());

    // 创建一个新的CUDA流，用于多个线程之间的同步
    let stream = cuda::Stream::new(device, cuda::StreamFlags::NON_BLOCKING).unwrap();

    // 在GPU上分配内存，并将其初始化为1
    let mut data = cuda::DeviceBuffer::from_slice(&[1], device);

    // 在GPU上运行一个简单的计算任务，将数据中的每一个元素都乘以2
    data.add_scalar(2, &stream).unwrap();

    // 将结果拷贝到CPU上的内存中
    let mut result = vec![0; 1];
    data.copy_to(&mut result[..], &stream).unwrap();

    // 打印结果
    println!("Result: {:?}", result);
}
