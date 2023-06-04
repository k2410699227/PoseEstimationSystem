
var domain = window.location.host;

new Vue({
    el: '#app',
    data: function () {
        return {
            imageUrl: "static/no-video.png",
            origin_width: 0,
            origin_height: 0,
            ws: null,
            video: null,
            canvas: null,
            capture: false,
            inputDevices: null,
            length: [{
                value: 128,
                label: '128'
            }, {
                value: 256,
                label: '256'
            }, {
                value: 512,
                label: '512'
            }, {
                value: 800,
                label: '800'
            }],
            marks: {
                20: '20%',
                40: '40%',
                60: '60%',
                80: '80%'
            },
            camera: null,
            picsize: 512,
            confident_threshold: 40,
            no_box: false,
            no_prob: false
        }
    },
    computed: {
        buttonText() {
            return !this.capture ? '开始' : '停止';
        },
        source() {
            if (this.camera === null) {
                return "未指定";
            }
            const index = this.inputDevices.findIndex(item => item.value === this.camera);
            return this.inputDevices[index].label;
        }
    },
    methods: {
        startBtnStatus() {
            if (this.capture) {
                return "danger"
            }
            else {
                return "primary"
            }
        },
        formatTooltip(val) {
            return val + "%";
        },
        startBtnAction() {
            if (this.camera == null) {
                this.$message({
                    message: '请先选择摄像头',
                    type: 'warning'
                });
                return
            }
            this.capture = !this.capture;
            this.onChangeDevice();
        },
        onChangeDevice() {
            console.log("device onload")
            var width, height
            if (!this.capture)
                return;
            for (let i = 0; i < this.inputDevices.length; i++) {
                if (this.inputDevices[i].value === this.camera) {
                    width = this.inputDevices[i].width;
                    height = this.inputDevices[i].height;
                    break;
                }
            }
            console.log("摄像头尺寸" + width + height)
            const constraints = {
                video: {
                    deviceId: this.camera
                }
            }
            // 获取摄像头并设置video元素
            navigator.mediaDevices.getUserMedia(constraints)
                .then(stream => {

                    this.video.srcObject = stream
                    this.video.play()

                }).catch(error => {
                    console.error(error);
                })
        }

    },

    mounted: function () {

        this.ws = io("ws://" + domain, { transports: ["websocket"] });

        this.ws.on("connect", () => {
            console.log(`已连接至${domain}`);

        });

        this.ws.on("connect_error", (error) => {
            console.log(`连接${domain}失败`);
        });
        getDevices().then((result) => {
            this.inputDevices = result; // 将解决值赋值给数据属性

        }).catch(error => {
            console.error(error);
            alert('无法获取摄像头信息');
            // this.$message.error('无法获取摄像头信息');
        });


        this.canvas = document.createElement('canvas')
        this.video = document.createElement('video')
        this.video.setAttribute("style", "display: none;")
        document.body.append(this.video)
        // this.video = this.$refs.video

        this.video.onloadedmetadata = () => {
            console.log("video onload")
            this.origin_width = this.video.videoWidth
            this.origin_height = this.video.videoHeight
            this.canvas.width = this.video.videoWidth
            this.canvas.height = this.video.videoHeight
        };



        // 将视频流渲染到canvas元素中
        // const canvas = this.$refs.canvas
        this.canvas.setAttribute("style", "display: none;")
        document.body.append(this.canvas)



        const drawFrame = () => {

            ctx = this.canvas.getContext('2d')

            // console.log(this.capture)
            if (!this.capture) {
                requestAnimationFrame(drawFrame)
                return;
            }
            ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height)


            this.canvas.toBlob((blob) => {
                var start = Date.now()
                emitWithTimeout(this.ws, "image", { data: blob, threshold: this.confident_threshold, no_box: this.no_box, no_prob: this.no_prob, short_edge: this.picsize }, 5000)
                    .then((response) => {
                        console.log(Date.now() - start + "ms")
                        // console.log(canvas.width, canvas.height, response)
                        resblob = new Blob([response.pose], { type: response.encoding });
                        url = URL.createObjectURL(resblob);
                        this.imageUrl = url

                        requestAnimationFrame(drawFrame)
                    })
                    .catch((error) => {
                        this.$message({
                            message: '请求超时',
                            type: 'warning'
                        });
                        if (!this.ws.connected) {
                            console.log(`尝试重连${domain}`);
                            this.ws.connect();
                        }

                        requestAnimationFrame(drawFrame)
                    });
            }, 'image/jpeg', 0.8);

        }
        requestAnimationFrame(drawFrame)
    },

})