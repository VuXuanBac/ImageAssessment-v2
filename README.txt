# Image Quality Assessment

## Overview

- Dựa trên mô hình NIMA do **Hossein Talebi** và **Peyman Milanfar** đề xuất: [Paper](https://arxiv.org/abs/1709.05424)
- Mục tiêu: Đánh giá chất lượng hình ảnh gần với đánh giá của con người
- Hướng tiếp cận: 
    - Xây dựng mô hình dựa trên các mô hình cho kết quả tốt trong bài toán phân loại hình ảnh với tập dữ liệu lớn như ImageNet. 
      - VD: VGG, Inception, MobileNet,... hay mới hơn là EfficientNet, ResNet, DenseNet,...
    - Thay vì chỉ học điểm trung bình hay đánh giá chất lượng theo hai mức cao/thấp, mô hình này thực hiện học **phân bố** của điểm đánh giá (VD: Có bao nhiêu người đánh giá $m$ điểm cho ảnh, với $m \in [1, 10]$)
      - Điều này giúp không chỉ đánh giá được tính thẩm mĩ của ảnh (điểm trung bình - mean) mà còn đánh giá được tính độc đáo của ảnh, thông qua độ lệch chuẩn (standard deviation) của phân bố. Ví dụ, với đề tài ảnh về một con tàu, một bức ảnh chụp về một con tàu vẽ trên giấy rõ ràng là độ thẩm mĩ không cao, song một số người sẵn sàng cho điểm cao vì sự sáng tạo, tính độc đáo của bức ảnh.
- Tập dữ liệu: [AVA Dataset](https://github.com/imfing/ava_downloader)
    - AVA Dataset với khoảng 255,000 hình ảnh.
    - Mỗi ảnh được đánh giá (cho điểm) bởi khoảng 200 thợ chụp ảnh nghiệp dư.
    - Dữ liệu cung cấp bao gồm *hình ảnh* và *số lượng người đánh giá từng mức điểm [1, 10]* cho từng ảnh.
      - Phân bố điểm trung bình là phân bố chuẩn, mean = 5.5
      - Phân bố độ lệch chuẩn của điểm là phân bố chuẩn, mean = 1.4

## Propose

- Phương pháp được đề xuất trong Paper cho kết quả tốt.
- Tuy nhiên, số lượng hình ảnh trong dải trung bình (4.5-6.5) chiếm áp đảo, gấp 10-11 lần hai dải thấp (< 4.5) và cao (> 6.5), nên gần như các đặc trưng mà mô hình học được là từ tập ảnh này, ngoài ra, tính độc đáo (thể hiện qua độ lệch chuẩn) cũng không học được.
- Do đó, project này thực hiện một số hướng phát triển sau:
  - Chia tập hình ảnh theo khoảng điểm đánh giá. VD: Ảnh có điểm đánh giá trung bình là từ 4.5 - 6.5
  - Sử dụng mô hình tương tự NIMA [Feature Learners] để học từng tập ảnh nhỏ.
  - Sau đó, có hai hướng kết hợp các mô hình này:
    - [1] Propose_1: Huấn luyện bộ kết hợp (ANN) [Combinator] để học cách kết hợp điểm đánh giá của các [Feature Learners].
      - Một ảnh mới sẽ được đánh giá qua các [Feature Learners], và [Combinator] sẽ kết hợp các kết quả đó lại để ra kết quả cuối cùng.
    - [2] Propose_2: Huấn luyện bộ phân loại (CNN) [Classifier] để học đặc trưng của ảnh và phân loại, sau đó sử dụng [Feature Learners] tương ứng để đánh giá.
      - Một ảnh mới sẽ được phân loại theo [Classifier] và kết quả đánh giá là của [Feature Learners] tương ứng.
      - Có thể kết hợp các [Feature Learners].

## Result

- Kết quả cho thấy cả hai mô hình đề xuất đều *KHÔNG* đem lại kết quả tốt hơn (nhưng cũng không tệ hơn) mô hình NIMA, mặc dù độ phức tạp mô hình là lớn hơn rất nhiều.
  - Có thể do bản chất của tập dữ liệu hoặc sự non nớt trong đề xuất ý tưởng hay sự thiếu chỉn chu trong triển khai mã nguồn.
- Tuy nhiên, đây là kết quả của quá trình học tập (Framework PyTorch, Assemble Learning, Imbalanced Handling...) và nghiên cứu (Hiểu Paper, Nhận ra vấn đề, Đề xuất giải pháp,...) của tôi, dưới sự hướng dẫn của thầy Trịnh Văn Chiến (SOICT - HUST)

## File Structure

- [data/](data/): Chứa các tệp xử lý dữ liệu (phân bố điểm)
- [src/](src/): Chứa các tệp mã nguồn huấn luyện và thử nghiệm.
- [result/](result/): Kết quả huấn luyện và thử nghiệm.
  - File [summary](result/summary.txt): Chứa kết quả so sánh giữa các mô hình.

- Notes:
  - Phiên bản (v) kí hiệu trong tên tệp chỉ xác định lần chạy và có thay đổi một số thông số của mô hình.
  - Các mô hình đề xuất cũng như là NIMA sử dụng BaseModel là: EfficientNet-B3.