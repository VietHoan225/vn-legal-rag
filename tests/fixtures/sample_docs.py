"""Sample legal documents fixture for unit testing."""

SAMPLE_LEGAL_DOC_1 = {
    "doc_id": 1,
    "title": "Văn bản số 99/2021/NĐ-CP",
    "category": "nđ-cp",
    "content": """VĂN BẢN SỐ 99/2021/NĐ-CP

Điều 1. Phạm vi điều chỉnh
Nghị định này quy định về quản lý hoạt động vận tải đường bộ. Căn cứ theo Điều 5 của Luật này, mọi hành vi vi phạm hành chính đều bị xử phạt nghiêm minh theo đúng trình tự pháp luật.

Điều 2. Đối tượng áp dụng
1. Tổ chức, cá nhân tham gia hoạt động vận tải đường bộ trên lãnh thổ Việt Nam.
2. Các cơ quan quản lý nhà nước có liên quan đến an toàn giao thông theo Điều 12 của Luật Giao thông đường bộ.

Điều 3. Quy định xử phạt chi tiết đối với hành vi vi phạm trong hoạt động vận tải đường bộ và bến xe
1. Phạt tiền từ 2.000.000 đồng đến 4.000.000 đồng đối với cá nhân, từ 4.000.000 đồng đến 8.000.000 đồng đối với tổ chức thực hiện hành vi vi phạm không niêm yết giá vé theo quy định hoặc niêm yết giá vé không đúng quy cách.
2. Phạt tiền từ 5.000.000 đồng đến 10.000.000 đồng đối với cá nhân, từ 10.000.000 đồng đến 20.000.000 đồng đối với tổ chức đón trả khách không đúng nơi quy định tại các trạm dừng đỗ hoặc trên các tuyến đường cấm.
3. Ngoài việc bị phạt tiền, cá nhân hoặc tổ chức vi phạm còn bị áp dụng hình thức xử phạt bổ sung bao gồm tước quyền sử dụng phù hiệu xe kinh doanh vận tải từ 01 tháng đến 03 tháng theo quy định tại Điều 8 Nghị định này.
4. Biện pháp khắc phục hậu quả buộc nộp lại số lợi bất hợp pháp có được do thực hiện hành vi vi phạm hành chính quy định tại các khoản 1, 2 Điều này vào ngân sách nhà nước.""",
}

SAMPLE_DOC_WITH_NO_ARTICLES = {
    "doc_id": 2,
    "title": "Văn bản số 00/2020/TB-VPCP",
    "category": "tb-vpcp",
    "content": "Thông báo kết luận của Thủ tướng Chính phủ tại cuộc họp phòng chống dịch bệnh.",
}

# Synthetic penalty article (structure modeled on traffic decrees, text written for tests).
SAMPLE_PENALTY_DOC = {
    "doc_id": 3,
    "title": "Văn bản số 01/2099/NĐ-CP",
    "category": "nđ-cp",
    "content": """Điều 1. Phạm vi điều chỉnh
Nghị định này quy định xử phạt vi phạm hành chính trong lĩnh vực giao thông thử nghiệm.

Điều 2. Xử phạt người điều khiển xe mô tô, xe gắn máy vi phạm quy tắc giao thông đường bộ
1. Phạt tiền từ 100.000 đồng đến 200.000 đồng đối với người điều khiển xe thực hiện một trong các hành vi vi phạm sau đây:
a) Không chấp hành hiệu lệnh, chỉ dẫn của biển báo hiệu, vạch kẻ đường;
b) Không có báo hiệu xin vượt trước khi vượt;
c) Không giữ khoảng cách an toàn để xảy ra va chạm với xe chạy liền trước;
d) Chuyển hướng không nhường quyền đi trước cho người đi bộ tại nơi có vạch kẻ đường;
đ) Chở người ngồi trên xe sử dụng ô (dù);
e) Điều khiển xe chạy dàn hàng ngang từ 03 xe trở lên.
2. Phạt tiền từ 200.000 đồng đến 300.000 đồng đối với người điều khiển xe thực hiện một trong các hành vi vi phạm sau đây:
a) Dừng xe, đỗ xe trên phần đường xe chạy ở đoạn đường ngoài đô thị nơi có lề đường;
b) Điều khiển xe không có còi hoặc có còi nhưng không có tác dụng;
c) Không đội mũ bảo hiểm cho người đi mô tô, xe máy khi điều khiển xe tham gia giao thông trên đường bộ;
d) Quay đầu xe tại nơi không được quay đầu xe.
3. Ngoài việc bị phạt tiền, người điều khiển xe còn bị tước quyền sử dụng giấy phép lái xe từ 01 tháng đến 03 tháng.

Điều 3. Hiệu lực thi hành
Nghị định này có hiệu lực từ ngày 01 tháng 01 năm 2099. Các quy định trước đây trái với Nghị định này đều bãi bỏ. Bộ trưởng các bộ, thủ trưởng cơ quan ngang bộ chịu trách nhiệm thi hành Nghị định này. Cơ quan, tổ chức và cá nhân có liên quan chịu trách nhiệm thi hành. Trong quá trình thực hiện, nếu có vướng mắc thì phản ánh kịp thời về Bộ Giao thông vận tải để hướng dẫn hoặc trình Chính phủ sửa đổi, bổ sung cho phù hợp với thực tế.""",
}
