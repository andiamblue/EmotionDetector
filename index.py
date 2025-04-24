# -*- coding: utf-8 -*- # این خط برای اطمینان از پشتیبانی کاراکترهای فارسی در کامنت‌ها خوب است

# کتابخانه‌های مورد نیاز را وارد می‌کنیم
import cv2
# from fer import FER # <<< این خط رو حذف یا کامنت کن (انجام شد)
from deepface import DeepFace # <<< استفاده از DeepFace
import time # (اختیاری) برای محاسبه FPS
import numpy as np # ممکن است برای برخی عملیات لازم شود

print("Libraries imported successfully.")

# فعال‌سازی دوربین (معمولاً 0 برای وبکم پیش‌فرض)
cap = cv2.VideoCapture(0)

# بررسی اینکه دوربین با موفقیت باز شده یا نه
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
else:
    print("Webcam opened successfully.")

# نیازی به مقداردهی اولیه آشکارساز جداگانه با DeepFace نیست
# detector = FER(mtcnn=True) # <<< حذف یا کامنت شد

print("Starting video stream processing...")

# (اختیاری) متغیرها برای محاسبه FPS
prev_frame_time = 0
# new_frame_time = 0 # مقداردهی اولیه داخل حلقه کافی است

while True:
    # خواندن یک فریم از دوربین
    ret, frame = cap.read()

    # اگر فریم با موفقیت گرفته نشد، از حلقه خارج شوید
    if not ret:
        print("Error: Failed to grab frame from webcam.")
        break

    # --- (اختیاری) محاسبه FPS ---
    new_frame_time = time.time()
    # جلوگیری از تقسیم بر صفر در فریم اول و مقداردهی اولیه prev_frame_time
    if prev_frame_time > 0:
        fps = 1 / (new_frame_time - prev_frame_time)
        fps_text = f"FPS: {int(fps)}"
        # استفاده از رنگ سبز روشن (BGR) برای متن FPS
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 0), 2)
    prev_frame_time = new_frame_time
    # --- پایان FPS اختیاری ---

    try:
        # --- استفاده از DeepFace برای تحلیل احساسات ---
        # enforce_detection=False: اگر چهره‌ای پیدا نشد، برنامه کرش نمی‌کند.
        # actions=['emotion']: فقط تحلیل احساسات را درخواست می‌کنیم.
        # detector_backend: می‌توانید 'opencv', 'ssd', 'mtcnn', 'dlib', 'retinaface' را امتحان کنید.
        # توجه: بار اول ممکن است نیاز به دانلود مدل‌ها باشد (نیاز به اینترنت).
        results = DeepFace.analyze(img_path=frame,
                                   actions=['emotion'],
                                   enforce_detection=False,
                                   detector_backend='opencv' # <<< این را می‌توانید تغییر دهید
                                  )

        # DeepFace لیستی از دیکشنری‌ها برمی‌گرداند، یکی برای هر چهره پیدا شده.
        # مطمئن می‌شویم که result از نوع لیست است (حتی اگر یک چهره باشد یا هیچ چهره‌ای نباشد)
        if isinstance(results, list) and len(results) > 0:
             # ممکن است بیش از یک چهره در تصویر باشد، روی همه نتایج پیمایش می‌کنیم
             for result in results:
                 # کلیدهای deepface کمی متفاوت است: 'region' برای کادر, 'dominant_emotion' برای احساس اصلی
                 # بررسی می‌کنیم که آیا کلیدهای مورد نیاز در نتیجه وجود دارند
                 if 'region' in result and 'dominant_emotion' in result and 'emotion' in result:
                    bounding_box = result['region']
                    dominant_emotion = result['dominant_emotion']
                    # نمره احساس غالب را از دیکشنری 'emotion' استخراج می‌کنیم
                    emotion_score = result['emotion'][dominant_emotion]

                    # مختصات کادر (x, y, w, h)
                    x = bounding_box['x']
                    y = bounding_box['y']
                    w = bounding_box['w']
                    h = bounding_box['h']

                    # رسم مستطیل سبز دور چهره
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # آماده‌سازی متن برای نمایش (احساس و نمره گرد شده به صورت درصد)
                    text = f"{dominant_emotion} ({round(emotion_score)}%)"

                    # قرار دادن متن آبی بالای مستطیل
                    # کمی تنظیم مکان متن برای خوانایی بهتر
                    text_y = y - 10 if y - 10 > 10 else y + 10 # جلوگیری از رفتن متن به خارج از کادر بالا
                    cv2.putText(frame, text, (x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                 # else:
                     # اگر کلیدهای لازم نبود، یعنی تحلیل کامل انجام نشده
                     # print("Analysis result dictionary missing expected keys.")
                     # pass

        # else:
             # اگر هیچ چهره‌ای پیدا نشد یا فرمت نتیجه غیرمنتظره بود
             # print("No face detected or analysis format unexpected in this frame.")
             # pass


    except Exception as e:
        # اگر در حین آنالیز خطایی رخ داد (مفید برای اشکال‌زدایی)
        # print(f"Error during DeepFace analysis: {e}") # <<< برای دیباگ، این خط را از کامنت خارج کنید
        pass # در حالت عادی، فقط از این فریم رد می‌شویم

    # نمایش فریم در یک پنجره با عنوان انگلیسی
    cv2.imshow("Emotion Detection (Press 'q' to exit)", frame)

    # بررسی برای فشار دادن دکمه 'q' برای خروج از حلقه
    # waitKey(1) منتظر 1 میلی‌ثانیه برای فشار کلید می‌ماند
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("'q' key pressed. Exiting...")
        break

# آزاد کردن منبع دوربین (بسیار مهم)
print("Releasing webcam resource...")
cap.release()

# بستن تمام پنجره‌های باز شده توسط OpenCV (بسیار مهم)
print("Closing OpenCV windows...")
cv2.destroyAllWindows()

print("Script finished.")