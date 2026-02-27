## Stream RTSP

In this project we take an RTSP stream (IP Cameras) run it through anonymization with YOLO and OpenCV then re-stream the output as RTSP.

We use an RTSP server which runs in a separate container on port `8554` and we utilize `opencv` to read the incoming RTSP stream and output a stream using `ffmpeg`

## How to Run

### 1. Create docker bridge network

```bash
docker network create rtspnetwork
```

### 2. Create docker stream container
```bash
docker build -t stream_rtsp .
```

### 3. Run the RTSP Server Container

```bash
docker run --rm -d --network=rtspnetwork --name rtsp_server -p 8554:8554 bluenviron/mediamtx:latest
```

### 4. Run the Program

```bash
docker run --rm -it --network=rtspnetwork -e STREAM_INPUT=rtsp://rtsp_server:8554/{stream_name} -e STREAM_OUTPUT=rtsp://rtsp_server:8554/output_{stream_name} {program container}
```

### 4.1 Run with Webcam
```bash
docker run --rm -it --network=rtspnetwork --device=/dev/video0:/dev/video0 -e STREAM_INPUT=0 -e STREAM_OUTPUT=rtsp://rtsp_server:8554/output_webcam stream_rtsp
```

---

**NOTE**:
Before running the inference the stream must be up and running.

In the example above the stream input is using the same `rtsp_server` however, that will be replaced with the RTSP address from the IP Cameras connected on the local network.

---

## Mimic an RTSP from a video (local)

### 1. Install ffmpeg

```bash
brew install ffmpeg
```

### 2. Run the stream on a video

This command assumes you are in a directory with a video named `video.mp4`

#### GPU + CUDA

```bash
ffmpeg -hwaccel cuda -re -stream_loop -1 -i video.mp4 -c:v h264_nvenc -preset p1 -tune ull -vf scale=1280:720 -b:v 1M -maxrate 1M -bufsize 3M -g 90 -keyint_min 45 -sc_threshold 0 -c:a copy -f rtsp -rtsp_transport tcp rtsp://localhost:8554/live
```

#### CPU

```bash
ffmpeg -re -stream_loop -1 -i video.mp4 -f rtsp -rtsp_transport tcp rtsp://localhost:8554/live
```

The server will be running on `rtsp://localhost:8554/live`
