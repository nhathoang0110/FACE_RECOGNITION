# FACE_RECOGNITION

# Main.py

# Những đường dẫn, tên folder cần lưu ý

+) Trong code: 

    +) "image_to_debug": Là tên folder  chứa ảnh nhân viên được checkin để debug. mỗi ngày sẽ có 1 folder, mỗi nhân viên sẽ có 1 ảnh. Chỉ cần tạo folder này, còn việc sinh ra folder mỗi ngày là tự động.

    +) "video_test.mp4" là tên video để test, nếu camera thì thay bằng camera

    +) "output.avi" là tên video đầu ra để demo cho thầy cô, không liên quan đến hệ thống, k cần tạo

    +) "results": là tên folder chứa kết quả của chấm công của công ty theo mỗi ngày. Đây là folder tổng. Mỗi ngày sẽ tự tạo folder, có 2 file morning.json và afternoon.json

+)  Các đường dẫn folder truyền vào hàm run:  

    +) path_detection:  là đường dẫn tới model detection

    +) path_vectori: là đường dẫn tới model vectori

    +) folder_vector: là đường dẫn tới folder gallery vector của nhân viên (database/vectors/)

    +) path_to_headpose: là đường dẫn tới model headpose estimation


# Gen_gallery
# Là File make_gallery.py để tạo vectors cho mỗi người

+) folder_nhanvienup: là đường dẫn tới folder chứa ảnh cuả nhân viên. Cái này khả năng đọc từ database. 
                        Ảnh của nhân viên mỗi người 1 folder trong folder đó. 
+) Make_gallery là hàm để sinh vector cho nhân viên. truyền tên vào là được.

+) Các đường dẫn truyền vào hàm :
    
    +) "folder_database": là đường dẫn tới folder chứa ảnh box và vector của nhân viên (database) bao gồm images và vectors. Cần phải tạo trước 2 folder images và vectors
    
    +) path_detection, path_vectori ý nghĩa và đường dẫn như bên trên.
    
    +) "folder_nhanvienup": đường dẫn đã giải thich bên trên

Kết quả sẽ sinh ra folder chứa vector của các nhân viên ở database/vectors/...