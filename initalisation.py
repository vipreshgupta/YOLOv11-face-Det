import cv2
from estimator import CrowdDensityEstimation



def main():
    estimator = CrowdDensityEstimation()  

    # Open video capture (0 for webcam, or video file path)
    cap = cv2.VideoCapture("e2.mp4")

    # Get video properties for output
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('crowd-density-estimation.mp4', 
                          fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame, density_info = estimator.process_frame(frame)
        estimator.display_output(processed_frame, density_info)  # Display
        out.write(processed_frame)  # Write output frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
