var domain = window.location.host;

new Vue({
    el: "#app",
    data: function () {
        return {
            imageUrl: null,
            resultUrl: null,
            ws: null,
            confident_threshold: 40,
            no_box: false,
            no_prob: false,
            source: null,
            result: null,
            info: null,
            encoding: 'jpg',
        }
    },
    computed: {
        source_name() {
            if (this.source === null)
                return '未选择';
            return this.source.name;
        },
        getImageUrl() {
            if (this.imageUrl === null)
                return "/static/upload.png";
            return this.imageUrl;
        },
        getResultUrl() {
            if (this.resultUrl === null)
                return "/static/no_image.png";
            return this.resultUrl;
        }
    },

    methods: {
        startUpload() {
            const fileInput = this.$refs.fileInput;
            // this.ws.emit("ping", "ping", (response) => {
            //     console.log(response)
            //   });
            fileInput.click()
        },
        handleFileSelect(event) {

            this.source = event.target.files[0];
            console.log(this.source)
            this.imageUrl = URL.createObjectURL(this.source);
            this.uploadFile();
        },
        uploadFile() {
            if (this.source === null)
                return;
            const loading = this.$loading({
                lock: true,
                text: '处理中',
                spinner: 'el-icon-loading',
                background: 'rgba(0, 0, 0, 0.7)'
            });
            const blob = new Blob([this.source], { type: this.source.type });
            emitWithTimeout(this.ws, "image", { data: blob, threshold: this.confident_threshold, no_box: this.no_box, no_prob: this.no_prob, short_edge: 512 }, 5000)
                .then((response) => {
                    console.log("收到回应")
                    console.log(response.instances)
                    this.info = response.instances;
                    // console.log(canvas.width, canvas.height, response)
                    resblob = new Blob([response.pose], { type: response.encoding });
                    url = URL.createObjectURL(resblob);
                    this.result = resblob;
                    this.resultUrl = url
                    this.encoding = response.encoding;
                    loading.close();
                })
                .catch((error) => {
                    this.$message({
                        message: '请求超时',
                        type: 'warning'
                    });
                    console.log("回应失败")
                    loading.close();
                    if (!this.ws.connected) {
                        console.log(`尝试重连${domain}`);
                        this.ws.connect();
                    }
                    console.error(error)
                });
        },
        downloadFile() {
            const link = document.createElement('a');
            // const blob = new Blob([this.result], { type: this.source.type });
            link.href = URL.createObjectURL(this.result);
            link.download = "res_" + this.source.name.substring(0, this.source.name.lastIndexOf("."))+this.encoding;

            // 触发链接的点击事件以开始下载
            link.click();
        },
        downloadJsonFile() {
            const jsonData = JSON.stringify(this.info);
            const blob = new Blob([jsonData], { type: 'application/json' });
            const fileUrl = URL.createObjectURL(blob);

            const link = document.createElement('a');
            link.href = fileUrl;
            link.download = this.source.name + ".json";

            // 触发链接的点击事件以开始下载
            link.click();
        },

        formatTooltip(val) {
            return val + "%";
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


    },

})