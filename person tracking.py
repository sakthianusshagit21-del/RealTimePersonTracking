import cv2
import numpy as np
import time

# -------------------------------
# Improved Centroid Tracker
# -------------------------------
class CentroidTracker:
    def __init__(self, max_disappeared=15):
        self.next_id = 0
        self.objects = {}       # id : (x, y, w, h, cx, cy)
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def register(self, box):
        x, y, w, h = box
        cx, cy = int(x + w/2), int(y + h/2)
        self.objects[self.next_id] = (x, y, w, h, cx, cy)
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def update(self, rects):
        if len(rects) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    del self.objects[obj_id]
                    del self.disappeared[obj_id]
            return self.objects

        input_centroids = []
        for (x, y, w, h) in rects:
            cx, cy = int(x + w/2), int(y + h/2)
            input_centroids.append((x, y, w, h, cx, cy))

        if len(self.objects) == 0:
            for item in input_centroids:
                self.register(item[:4])
            return self.objects

        used_ids = set()
        for (x, y, w, h, cx, cy) in input_centroids:
            distances = {
                obj_id: np.hypot(cx - ox, cy - oy)
                for obj_id, (_, _, _, _, ox, oy) in self.objects.items()
            }

            closest_id = min(distances, key=distances.get)
            if distances[closest_id] < 100:
                self.objects[closest_id] = (x, y, w, h, cx, cy)
                self.disappeared[closest_id] = 0
                used_ids.add(closest_id)
            else:
                self.register((x, y, w, h))

        for obj_id in list(self.objects.keys()):
            if obj_id not in used_ids:
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    del self.objects[obj_id]
                    del self.disappeared[obj_id]

        return self.objects

# -------------------------------
# Main Program
# -------------------------------
def main():
    cap = cv2.VideoCapture(0)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    tracker = CentroidTracker()
    frame_skip = 2
    count = 0
    detections = []
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        if count % frame_skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections, _ = hog.detectMultiScale(
                gray,
                winStride=(8, 8),
                padding=(8, 8),
                scale=1.03
            )
            print(f"People detected: {len(detections)}")

        objects = tracker.update(detections)

        for obj_id, (x, y, w, h, cx, cy) in objects.items():
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {obj_id}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow("Assignment 1 - Person Detection & Tracking", frame)
        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
