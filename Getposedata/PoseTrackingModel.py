import cv2
import time
import mediapipe as mp
from tqdm import tqdm  # import progress bar


class PoseDetector():
    def __init__(self, model=False, model_comple=1, smooth_lm=True, segmentation=True, smooth_segmentation=True,
                 detectionCon=0.5, trackCon=0.5):
        self.model = model
        self.model_comple = model_comple
        self.smooth_lm = smooth_lm
        self.segmentation = segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mppose = mp.solutions.pose
        self.Pose = self.mppose.Pose(self.model, self.model_comple, self.smooth_lm, self.segmentation,
                                     self.smooth_segmentation, self.detectionCon, self.trackCon)
        self.mpdraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.Pose.process(imgRGB)

        if self.results.pose_landmarks:
            self.mpdraw.draw_landmarks(img, self.results.pose_landmarks, self.mppose.POSE_CONNECTIONS)

        return img

    def findPosition(self, img):

        Plmlist = []
        pose2D = []
        pose3D = []
        xx = []
        yy = []
        zz = []
        DataPose = [[], [], []]
        if self.results.pose_landmarks:
            myposeLms = self.results.pose_landmarks
            for id, lm in enumerate(myposeLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), lm.z
                # print(id, ':', cx, cy)
                # if id == 26:
                #     cv2.circle(img, (cx, cy), 10, (0, 255, 255), cv2.FILLED)
                Plmlist.append([id, cx, cy, cz])
                pose2D.append([lm.x, lm.y])
                pose3D.append([lm.x*640, lm.y*480, lm.z])
                xx.append(lm.x)
                yy.append(lm.y)
                zz.append(lm.z)
                DataPose[0] = xx
                DataPose[1] = yy
                DataPose[2] = zz
        return Plmlist, pose2D, pose3D, DataPose

    def findPose_video(self, input_path='/home/xrf/PycharmProjects/gaitDetection/videos/result.mp4'):
        filehead = input_path.split('/')[-1]
        out_path = "out-" + filehead
        print("The video is processing...", input_path)

        # get total frames of video
        cap = cv2.VideoCapture(input_path)
        frame_count = 0
        while (cap.isOpened()):
            success, frame = cap.read()
            frame_count += 1
            if not success:
                break
        cap.release()
        print("The total frames is :", frame_count)

        cap = cv2.VideoCapture(input_path)
        frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(out_path, fourcc, fps, (int(frame_size[0]), int(frame_size[1])))

        # process bar connect with video total frames
        with tqdm(total=frame_count - 1) as pbar:
            try:
                while (cap.isOpened()):
                    success, frame = cap.read()
                    if not success:
                        break
                    try:
                        frame = self.findPose(frame)
                    except:
                        print("error")
                        pass
                    if success == True:
                        out.write(frame)
                        # process bar update one frame
                        pbar.update(1)
            except:
                print("interrupt in the middle of process ")
                pass
        out.release()
        cap.release()
        print("The video has been saved", out_path)


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        img2 = detector.findPose(img)
        Plmlist = detector.findPosition(img2, True)

        if len(Plmlist) != 0:
            print(Plmlist[10])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img2, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
    cap.release()


if __name__ == "__main__":
    main()
