import os
import cv2
import ffmpeg


def stream_rtsp(input_url, output_url):
    print(f"Input URL: {input_url}")
    print(f"Output URL: {output_url}")

    cap = cv2.VideoCapture(input_url, cv2.CAP_DSHOW)
    

    if not cap.isOpened():
        print(f"Error opening video stream: {input_url}")
        return

    # Retrieve frame width, height, and fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    process = (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="bgr24",
            s="{}x{}".format(width, height),
            framerate=fps,
        )
        .output(
            output_url, format="rtsp", rtsp_transport="tcp", vcodec="libx264", r=fps
        )
        .global_args("-re")  # Stream in real-time
        .run_async(pipe_stdin=True)
    )
    retries = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if retries < 10:
                    print("Failed to grab frame, retrying...")
                    retries += 1
                    continue
                print("Error reading frame. Stream may have ended or corrupted.")
                break

            try:
                process.stdin.write(frame.tobytes())
            except IOError as e:
                print(f"I/O Error: {e}")
                break

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        cap.release()
        process.stdin.close()
        process.wait()    


if __name__ == "__main__":
    input_url = os.getenv("STREAM_INPUT")
    output_url = os.getenv("STREAM_OUTPUT")
    if not output_url:
        output_url = "rtsp://localhost:8554/stream"
    if input_url and output_url:
        stream_rtsp(input_url, output_url)
    elif output_url:
        # if no input URL is provided, stream from the default webcam
        stream_rtsp(0, output_url)
        
    
