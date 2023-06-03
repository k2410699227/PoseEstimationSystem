function emitWithTimeout(socket, event, data, timeout) {
    return new Promise((resolve, reject) => {
      let isResolved = false;

      // 发送事件
      socket.emit(event, data, (response) => {
        if (!isResolved) {
          // 收到响应，标记为已解决
          isResolved = true;
          resolve(response);
        }
      });

      // 设置超时定时器
      setTimeout(() => {
        if (!isResolved) {
          // 超时未收到响应，标记为已解决，并拒绝 Promise
          isResolved = true;
          reject(new Error('Emit timeout'));
        }
      }, timeout);
    });
  }

// 列举可用的媒体输入设备
async function getDevices() {
  var result = []

  devices = await navigator.mediaDevices.enumerateDevices();
  // console.log(devices)
  videoDevices = devices.filter(device => device.kind === 'videoinput');
  videoDevices.forEach(device=>{
    let res = {}
    constraints = {
      video: {
          deviceId: device.deviceId
      }
  }
  let width = 0 ,height = 0
  // 获取摄像头并设置video元素
  navigator.mediaDevices.getUserMedia(constraints)
      .then(stream => {

        const track = stream.getVideoTracks()[0];
        const capabilities = track.getCapabilities();
        width = capabilities.width.max
        height = capabilities.height.max

      }).catch(error => {
          console.error(error);
      })
    res.value = device.deviceId;
    res.label = device.label;
    res.width = width;
    res.height = height;
    result.push(res)})
  return result;
}