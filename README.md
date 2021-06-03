# FACE_RECOGNITION

# Main.py

# Những đường dẫn, tên folder cần lưu ý

+) Trong code: 

    +) "image_to_debug": CẦN TẠO: Là tên folder  chứa ảnh người lạ để debug. mỗi ngày sẽ có 1 folder, mỗi người lạ sẽ có 1 ảnh. Chỉ cần tạo folder này, còn việc sinh ra folder mỗi ngày là tự động. Tên ảnh sẽ có dạng datatime_random_flag, trong đó flag=0 là sáng, còn bằng 1 là chiều. (ví dụ 1:30:7_1_0.jpg)

    +) "video_test.mp4" là tên video để test, nếu camera thì thay bằng camera

    +) "output.avi" là tên video đầu ra để demo cho thầy cô, không liên quan đến hệ thống, k cần tạo

    +) "results": CẦN TẠO: là tên folder chứa kết quả của chấm công của công ty theo mỗi ngày. Đây là folder tổng. Mỗi ngày sẽ tự tạo folder, có 2 file morning.json và afternoon.json

+)  Các biến, đường dẫn folder truyền vào hàm run:  

    +) path_detection:  là đường dẫn tới model detection

    +) path_vectori:   là đường dẫn tới model vectori

    +) folder_vector: là đường dẫn tới folder gallery vector của nhân viên (database/vectors/)

    +) path_to_headpose:  là đường dẫn tới model headpose estimation

    +) flag: là biến chỉ sáng hay chiều, nếu 0 là sáng còn 1 là chiều

    +) type_cam: Biến chỉ camera front hay xa, cố định 0. 1 là để demo vs thầy cô thêm về camera xa.


# Gen_gallery
# Là File make_gallery.py để tạo vectors cho mỗi người

+) folder_nhanvienup: CẦN TẠO là đường dẫn tới folder chứa ảnh cuả nhân viên. Cái này khả năng đọc từ database. 
                        Ảnh của nhân viên mỗi người 1 folder trong folder đó. 
+) make_gallery là file chứa hàm để sinh vector cho nhân viên. truyền tên vào là được.

+) Các đường dẫn truyền vào hàm :
    
    +) "folder_database": là đường dẫn tới folder chứa ảnh box và vector của nhân viên (database) bao gồm images và vectors. 
                          CẦN TẠO trước 2 folder images và vectors
    
    +) path_detection, path_vectori ý nghĩa và đường dẫn như bên trên.
    
    +) "folder_nhanvienup": đường dẫn đã giải thich bên trên

    +) "name" là biến tên nhân viên chắc nhận từ frontend. Cái này chưa rõ ấn cho từng người hay ấn cho cả nên đang để là ấn cho từng người.
            Khả năng là admin sẽ ấn cho từng người, vì admin phải check xem bọn nhân viên up ảnh có ok không hay là up vớ vẩn

Kết quả sẽ sinh ra folder chứa vector của các nhân viên ở database/vectors/...