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

<table>
  <tr>
    <td><img src="https://arxiv.org/html/2410.24159v2/x1.png" width="50%"/></td>
    <td width="150%"><img src="https://private-user-images.githubusercontent.com/12207571/319390512-48efd48a-431b-4625-8e0f-248a442e3839.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyNjYxNzEsIm5iZiI6MTc0ODI2NTg3MSwicGF0aCI6Ii8xMjIwNzU3MS8zMTkzOTA1MTItNDhlZmQ0OGEtNDMxYi00NjI1LThlMGYtMjQ4YTQ0MmUzODM5LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDEzMjQzMVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTEzOTk1NWY4MWU5N2E3NWM1MjExMzkwZjNiMjlhMzQ5OWM1NzlhMGRhYzI4MmMwZmI5NWZhYzQ0ZGI2OThhMzkmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.GXRq3cQnXiyD_vYF-4O9HagAM3Fcug7rxjtr8fvGoXU"/></td>
  </tr>
</table>

## Byte Latent Transformer
![](https://arxiv.org/html/2412.09871v1/extracted/6066458/assets/patching_types.png)
![](https://arxiv.org/html/2412.09871v1/x4.png)
![](https://arxiv.org/html/2412.09871v1/x5.png)
![](https://arxiv.org/html/2412.09871v1/x3.png)
<!-- ![](https://arxiv.org/html/2412.09871v1/extracted/6066458/assets/patching.png) -->

## vec2vec: translate text embeddings across different spaces without any paired data or encoders
- https://x.com/rishi_d_jha/status/1925212069168910340
- **https://x.com/jxmnop/status/1925224618060587523**

## mixture of tokenizers
- https://x.com/omouamoua/status/1922934072730403228
- https://github.com/snimu/blog/tree/main/contents/mixture-of-tokenizers
- https://github.com/snimu/blog/blob/main/contents/mixture-of-tokenizers-math/article.md
- https://github.com/snimu/blog/blob/main/contents/mot-scaling/article.md
