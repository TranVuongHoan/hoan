# top 5 quoc gia tiem vaccine nhieu nhat
import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ tệp Excel
df = pd.read_csv('C:/Users/thengoc/Downloads/vaccination-data.csv')

# Sắp xếp dữ liệu theo cột "Total_Vaccinations" giảm dần
df_sorted = df.sort_values('TOTAL_VACCINATIONS', ascending=False)

# Lấy ra 10 quốc gia có số lượng tiêm vaccine nhiều nhất
top_countries = df_sorted.head(10)

# Tạo biểu đồ cột cho 10 quốc gia nhiều nhất

plt.bar(top_countries['COUNTRY'], top_countries['TOTAL_VACCINATIONS'])

# Đặt tên cho trục x và trục y
plt.xlabel('Quốc gia')
plt.ylabel('Số lượng tiêm vaccine')

# Xoay nhãn trục x nếu cần thiết
plt.xticks(rotation=45, fontsize= 8)

# Hiển thị biểu đồ
plt.show()

# TI LE TIEM VACCINE /100 DÂN\

import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ tệp Excel
df = pd.read_csv('C:/Users/thengoc/Downloads/vaccination-data.csv')

# Sắp xếp dữ liệu theo cột "TOTAL_VACCINATIONS_PER100" giảm dần
df_sorted = df.sort_values('TOTAL_VACCINATIONS_PER100', ascending=False)
 # Lấy ra top 5 quốc gia có tỷ lệ tiêm vaccine cao nhất
top_countries = df_sorted.head(5)
# Tạo biểu đồ cột
plt.bar(top_countries['COUNTRY'], top_countries['TOTAL_VACCINATIONS_PER100'])
# Đặt tên cho trục x và trục y
plt.xlabel('Quốc gia')
plt.ylabel('Tỷ lệ tiêm vaccine (số lần/100 dân)')

# Xoay nhãn trục x nếu cần thiết
plt.xticks(rotation=45)

# Hiển thị biểu đồ
plt.show()

#số lượng ít nhất 1 người tiêm vaccine top 5 quốc gia
import pandas as pd
import matplotlib.pyplot as plt
# Đọc dữ liệu từ tệp Excel
df = pd.read_csv('C:/Users/thengoc/Downloads/vaccination-data.csv')
# Sắp xếp dữ liệu theo cột "TOTAL_VACCINATIONS_1plus_dose" giảm dần
df_sorted = df.sort_values('PERSONS_VACCINATED_1PLUS_DOSE', ascending=False)
# Lấy ra top 5 quốc gia có số lượng tiêm vaccine (1+ dose) cao nhất
top_countries = df_sorted.head(5)
# Tạo biểu đồ cột
plt.bar(top_countries['COUNTRY'], top_countries['PERSONS_VACCINATED_1PLUS_DOSE'])
# Đặt tên cho trục x và trục y
plt.xlabel('Quốc gia')
plt.ylabel('Số lượng tiêm vaccine (1+ dose)')
# Xoay nhãn trục x nếu cần thiết
plt.xticks(rotation=45, fontsize= 8)
# Hiển thị biểu đồ
plt.show()

#

import pandas as pd
data= pd.read_csv("C:/Users/thengoc/Downloads/vaccination-data.csv")
data["FIRST_VACCINE_DATE"]= pd.to_datetime(data["FIRST_VACCINE_DATE"], format = "%d/%m/%Y")
sorted_data= data.sort_values("FIRST_VACCINE_DATE")
top_10_countries= sorted_data.head(10)["COUNTRY"]
print(top_10_countries)

#

import pandas as pd
data= pd.read_csv("C:/Users/thengoc/Downloads/vaccination-data.csv")
data["FIRST_VACCINE_DATE"]= pd.to_datetime(data["FIRST_VACCINE_DATE"], format = "%d/%m/%Y")
sorted_data= data.sort_values("FIRST_VACCINE_DATE")
top_10_countries= sorted_data.tail(10)["COUNTRY"]
print(top_10_countries)

#Tỷ lệ người đã tiêm ít nhất 1 liều vaccine theo khu vực WHO
import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file hoặc database

data = pd.read_csv("C:/Users/thengoc/Downloads/vaccination-data.csv") # Thay đổi "du_lieu.csv" bằng đường dẫn tới file dữ liệu thực tế của bạn

 # Vẽ biểu đồ cột
plt.figure(figsize=(10, 6))
plt.bar(data["WHO_REGION"], data["PERSONS_VACCINATED_1PLUS_DOSE_PER100"])
plt.xlabel("Khu vực WHO")
plt.ylabel("Tỷ lệ người đã tiêm ít nhất 1 liều vaccine (%)")
plt.title("Tỷ lệ người đã tiêm ít nhất 1 liều vaccine theo khu vực WHO")
plt.xticks(rotation=45)
plt.show()

#Phân phối loại vaccine (NUMBER_VACCINER_TYPE_USED) được sử dụng.
import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file hoặc database
data = pd.read_csv("C:/Users/thengoc/Downloads/vaccination-data.csv") # Thay đổi "du_lieu.csv" bằng đường dẫn tới file dữ liệu thực tế của bạn

# Tính tổng số lượng vaccine theo loại
vaccine_counts = data["NUMBER_VACCINES_TYPES_USED"].value_counts()

# Vẽ biểu đồ vòng
plt.figure(figsize=(8, 8))
plt.pie(vaccine_counts, labels=vaccine_counts.index, autopct='%1.1f%%')
plt.title("Phân phối loại vaccine")
plt.show()

#Tổng số liều vaccine được sử dụng (TOTAL_VACCINATIONS) theo ngày cập nhật (DATE_UPDATED).
import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file hoặc database
data = pd.read_csv("c:\Users\thengoc\Downloads\vaccination-data.csv") # Thay đổi "du_lieu.csv" bằng đường dẫn tới file dữ liệu thực tế của bạn

# Chuyển đổi cột "DATE_UPDATED" thành định dạng ngày
data["DATE_UPDATED"] = pd.to_datetime(data["DATE_UPDATED"], format="%d/%m/%Y")

# Tính tổng số liều vaccine theo ngày cập nhật
daily_vaccinations = data.groupby("DATE_UPDATED")["TOTAL_VACCINATIONS"].sum()

# Tạo biểu đồ đường
plt.figure(figsize=(12, 6))
plt.plot(daily_vaccinations.index, daily_vaccinations.values)
plt.xlabel("Ngày cập nhật")
plt.ylabel("Tổng số liều vaccine")
plt.title("Tổng số liều vaccine được sử dụng theo ngày cập nhật")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

#top 10 cty sử dụng vaccine cao nhất
import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file hoặc database
data = pd.read_csv("C:/Users/thengoc/Downloads/vaccination-metadata.csv") # Thay đổi "du_lieu.csv" bằng đường dẫn tới file dữ liệu thực tế của bạn
 # Tính số lượng vaccine theo từng công ty
vaccine_counts = data.groupby("COMPANY_NAME")["VACCINE_NAME"].count().sort_values(ascending=False).head(10)
 # Tạo biểu đồ cột
plt.figure(figsize=(10, 6))
vaccine_counts.plot(kind="bar")
plt.xlabel("Công ty")
plt.ylabel("Số lượng vaccine")
plt.title("Top 10 công ty có số lượng vaccine sử dụng cao nhất")
plt.xticks(rotation=45)
plt.show()

#top 10 sản phẩm có số lượng vaccine sử dụng cao nhất

import pandas as pd
import matplotlib.pyplot as plt
# Đọc dữ liệu từ file hoặc database
data = pd.read_csv("C:/Users/thengoc/Downloads/vaccination-metadata.csv") # Thay đổi "du_lieu.csv" bằng đường dẫn tới file dữ liệu thực tế của bạn
# Tính số lượng vaccine theo từng sản phẩm
vaccine_counts = data.groupby("PRODUCT_NAME")["VACCINE_NAME"].count().sort_values(ascending=False).head(10)
# Tạo biểu đồ cột
plt.figure(figsize=(10, 6))
vaccine_counts.plot(kind="bar")
plt.xlabel("Sản phẩm")
plt.ylabel("Số lượng vaccine")
plt.title("Top 10 sản phẩm có số lượng vaccine sử dụng cao nhất")
plt.xticks(rotation=45)
plt.show()

#Phân phối công ty (COMPANY_NAME) cung cấp vaccine.
import pandas as pd
import matplotlib.pyplot as plt
# Đọc dữ liệu từ file hoặc database
data = pd.read_csv("C:/Users/thengoc/Downloads/vaccination-metadata.csv") # Thay đổi "du_lieu.csv" bằng đường dẫn tới file dữ liệu thực tế của bạn

# Tính số lượng vaccine theo từng công ty
company_counts = data["COMPANY_NAME"].value_counts()
# Lấy danh sách các công ty và số lượng vaccine tương ứng
companies = company_counts.index.tolist()
vaccine_counts = company_counts.values.tolist()
# Tạo biểu đồ tròn
plt.figure(figsize=(8, 8))
plt.pie(vaccine_counts, labels=companies, autopct="%1.1f%%")
plt.title("Phân phối công ty cung cấp vaccine")
plt.show()

#Số lượng vaccine (VACCINE_NAME) được sử dụng theo từng sản phẩm (PRODUCT_NAME), xếp chồng theo công ty (COMPANY_NAME).

import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file hoặc database
data = pd.read_csv("C:/Users/thengoc/Downloads/vaccination-metadata.csv") # Thay đổi "du_lieu.csv" bằng đường dẫn tới file dữ liệu thực tế của bạn
# Tạo bảng pivot để tính số lượng vaccine theo từng sản phẩm và công ty
pivot_table = data.pivot_table(index="PRODUCT_NAME", columns="COMPANY_NAME", values="VACCINE_NAME", aggfunc="count")
# Tạo biểu đồ xếp chồng cột
plt.figure(figsize=(10, 6))
pivot_table.plot(kind="bar", stacked=True)
plt.xlabel("Sản phẩm")
plt.ylabel("Số lượng vaccine")
plt.title("Số lượng vaccine được sử dụng theo từng sản phẩm, xếp chồng theo công ty")
plt.xticks(rotation=45)
plt.legend(title="Công ty")

