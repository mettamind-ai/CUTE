# Super Token
1 visual token có rất nhiều thông tin (diễn đạt bằng nhiều text token)
Liệu có thể làm tương tự như visual encoder nhưng mà cho text? => **SUPER TEXT TOKEN**
1 cụm n text tokens giờ đc biểu diễn = 1 embedding vector thay vì n như trước =>
đó là biên giới hạn của ngôn ngữ con người.

Máy có thể khai thác cái này để thông minh hơn ví dụ: Con lai của hổ và sư tử miêu tả là 
`"1 con gì đó giống con hổ và con sư tử"` => từ mới `hổ sư` thay vì biểu diễn bằng 1 câu 
sẽ có 1 miền trong token embeddings biểu diễn cái này
miền đó nằm giữa trung điểm của vector hổ và vector sư tử

biển diễn cho input và ouput thì vẫn là tokens trong vocab (con chữ) 
để đảm bảo tính cross entropy loss như bình thường, 
nhưng khi vào model mình có thể **map con chữ thành concept vector** ...


|![](https://arxiv.org/html/2410.24159v2/x1.png)|![](https://arxiv.org/html/2404.05961v2/x2.png)|
|-|-|