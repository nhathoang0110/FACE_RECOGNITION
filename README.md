# FACE_RECOGNITION

# Main.py

+) Trong code: 

    +) Folder image_to_debug chứa ảnh nhân viên được checkin để debug. mỗi ngày sẽ có 1 folder, mỗi nhân viên sẽ có 1 ảnh.

    +) video_test.mp4 là video để test

    +) output.avi là video đầu ra để demo

    +) results là folder chứa kết quả của chấm công của công ty theo mỗi ngày. Mỗi ngày sẽ có 2 file morning và afternoon.

+) truyền vào hàm run:

    +) path_detection là link tới model detection

    +) path_vectori là link tới model vectori

    +) folder_vector là link tới folder gallery vector của nhân viên

    +) path_to_headpose là link tới folder headpose


# Gen_gallery

+) Ảnh của nhân viên mỗi người 1 folder trong folder folder_nhanvienup

+) Chạy file make_gallery

+) folder_database là folder chứa ảnh box và vector của nhân viên (database) bao gồm images và vectors

Kết quả sẽ sinh ra folder chứa vector của các nhân viên ở database/vectors/...