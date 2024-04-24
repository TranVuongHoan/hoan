#string: +
#numeric : + - * / % //
#a+=5
#while

"""number_1= 10
number_2= 10.5
string_1 = "abc"
boolean=True
list_1=[number_1, number_2, string_1, boolean]
number_1_str= str(number_1)
print(string_1 +str(number_1))
print(type(number_2))
print(type(number_1_str))"""

"""
    if condition:

    elif condition1:

    else:

"""

#break, continue

"""for i in range(10):
    print(i)
    if i> 5:
        break"""

#range(n) -> list 0 -> n-1
#range(1, n) -> list 1-> n-1
#range(0, n, 2)

"""for i in range(10):
    print(i)
    if i> 5:
        break

print(f"after i:{i}") #den 6 thi dung vong lap"""

"""for i in range(10):
    print(i)
if i> 5:
    continue

print(f"after i:{i}") #tiep tuc vong for nhung sau continue thi khong chay nua"""

"""string=input("Moi ban nhap string:")
num_vower= 0
for char in string:
    if char == "a":
        num_vower += 1
    elif char == "e":
        num_vower += 1
    elif char == "o":
        num_vower += 1
    elif char== "u":
        num_vower += 1
    elif char == "i" :
        num_vower +=1
    else:
        continue
print(num_vower)"""

# and , or

"""string=input("Moi ban nhap string:")
num_vower= 0

for char in string:
    if char == "a" or char == "o" or char == "u" or char == "i" or char == "e":
        num_vower += 1
    else:
        continue

print(num_vower)"""

"""string=input("Moi ban nhap string:")
num_vower= 0
for char in string:
    if char in "aeoui":
        num_vower += 1
    else:
        continue

print(num_vower)"""

"""string= int(input("nhap 1 so:")) #doi input thanh dang int khi can nhap 1 so
for i in range(1, string):
    if i % 2 == 1:
        print(i)
    else:
        continue"""

string= input("moi ban nhap 1 cau:")

#tách chuỗi thành các phần tử riêng biệt dựa trên khoảng trắng

for i in string.split():
    print(i)

string= input("moi ban nhap 1 tu:")

#trả về một đối tượng liệt kê (iterator) chứa các cặp giá trị (index, value)

for position, character in enumerate(string):
    print(f"Character: {character} - Position: {position}")

import random
# tạo ra một số ngẫu nhiên trong khoảng từ 1 đến 100.
random_number= random.randint(1, 100)
while True:
    guess=int(input("nhap 1 so ngau nhien giua 1 va 100:"))

    if guess == random_number:
        print("chuc mung doan dung")

    elif guess > random_number:
        print("higher")

    elif guess < random_number:
        print("lower")

    else:
        continue

"""list=[.....]
target_name= input("Nhap ten can tim:")
for i in names:
    if i == target_name:
        print("ten co xuat hien trong danh sach")
        break
    else:
        print("ten khong xuat hien")"""

"""number_input= input("nhap cac so voi dau cach:")
number_list= number_input.split()
number = list(number_list)
sum = 0
for i in number:
    sum += int(i)
print("tong cua day so la:", sum)"""

#------------------------------------------------------
# for position, character in...
#dung enumerate de tim position la index, character la phan tu
#type -> tim kieu du lieu
#list: append, sum

string= input("nhap 1 chuoi:")
list_str= string.split()
list_number= []

#Với mỗi lần lặp, giá trị của phần tử hiện tại trong list_str được gán cho biến string
for string in list_str:
    list_number.append(int(string))
    print(sum(list_number))

"""l= []
for i in range(10):
l.append(i+2)"""

"""l =[i+2 for i in range(10)]"""
"""string= input("nhap 1 chuoi:")
list_str= string.split()"""

"""list_number= []
for string in list_str:
list_number.append(int(string))

#list comprehension ->>> list_number =[int(string) for string in list_str]
print(sum(list_number))"""

#random

"""import random
int_random = random.randint(1, 100)
while True:
    number= int(input("nhap so:"))
    if number >int_random:
        print("lower")
    elif number < int_random:
        print("higher")
    else:
        print("correct")
        break"""

"""list_names=["a", "b", "c", "c", "d"]
name="a"
if name in list_names:
    print("name in list")
else:
    print("name not in list")"""

"""number= int(input("nhap 1 so:"))
if number < 0 or number > 100:
    print("impossible")
elif 49 >= number >= 0:
    print("fail")
elif 59 >= number >= 50:
    print("1")
elif 69 >= number >= 60:
    print("2")
elif 79>= number >= 70:
    print("3")
elif 89 >= number >= 80:
    print("4")
elif 100 >= number >= 90:
    print("5")"""

"""string= input("nhap 1 string:")
a= 20-len(string)
print(a*"*" + string)"""

#slicing
"""s[a:b:c]
a, b la index c la step"""

"""vi du s[0:5:2]"""

"""string= input("nhap 1 string:")
sub= input("nhap 1 substring")
index_1= s.index(sub)
s_1= s[index_1+ len(sub):]
index= s_1.index(sub) + index_1 +len(sub)"""

"""password =input("nhap password:")
length_check = len(password) >=8
uppercase_check =False
lowercase_check= False
digit_check = False
for char in password:
    if char.isupper():
        uppercase_check =True
    if char.islower():
        lowercase_check= True
    if char.isdigit():
        digit_check = True
    if length_check and uppercase_check and lowercase_check and digit_check:
        print("Strong password")
    else:
        print("weak password")"""

"""year = int(input("nhap 1 nam:"))
if year % 4 == 0:
    print("this is a leap year")
elif year % 100 == 0 and year % 400 == 0:
    print("this is a leap year")
else:
    print("this isn't a leap year")"""

"""string =input("nhap 1 chuoi:")
len_of_string = len(string)
is_enough=False
if len_of_string >=8:
    is_enough=True
    is_container_uppercase=False
    is_container_lowercase=False
    is_container_digit =False

for character in string:
if character.isupper():
    is_container_uppercase =True
if character.islower():
    is_container_lowercase =True
if character.isdigit():
    is_container_digit =True
if is_enough and is_container_uppercase and is_container_lowercase and is_container_digit:
    print("strong password")
else:
    print("weak password")"""

"""def function(string, number =10):
print("input function:", string, number)
return "hello" + string + str(number)"""

"""def check_leap-year(year):
if year % 4 == 0 or (year% 100== 0 and year%400 == 0):
    return "leap year"
else:
    return "not leap year"""

"""def square(x):
    return x ** 2
def triple(y):
    return y * 3
for i in range(1, 11):
    print(f"triple({i})== {triple(i)} square({i})== {square(i)}")"""

"""def rectangle(h, d):
    return h * d
def triangle(h, d):
    return h * d / 2
def circle(r):
    return r ** 2 * 3.14
def main(type_str, h=10, d=10, r=10):
if type_str =="rectangle":
    return f"area of{type_str} is {rectangle(h, d)}"
elif type_str =="triangle":
    return f"area of{type_str} is {triangle(h, d)}"
elif type_str =="circle":
    return f"area of{type_str} is {circle(r)}"
else:
    return "unknown"
print(main{"rectangle", 20, 10})"""

"""def calculate_triangle_area():
    base = float(input("Enter the base length: "))
    height = float(input("Enter the height: "))
    area = 0.5 * base * height
    print(f"The area of the triangle is: {area}")
def calculate_rectangle_area():
    length = float(input("Enter the length: "))
    width = float(input("Enter the width: "))
    area = length * width
    print(f"The area of the rectangle is: {area}")
def calculate_circle_area():
    radius = float(input("Enter the radius: "))
    area = 3.14 * radius ** 2
    print(f"The area of the circle is: {area}")

while True:
    shape = input("Enter the shape (triangle, rectangle, circle): ")
    if shape == "triangle":
        calculate_triangle_area()
    elif shape == "rectangle":
        calculate_rectangle_area()
    elif shape == "circle":
        calculate_circle_area()
    else:
        print("Unknown shape!")"""

"""def chessboard(length):
    for i in range(length):
    for j in range(length):
        if (i + j) % 2 == 0:
            print("1", end ="")
        else:
        print("0", end="")

        print()"""

"""def squared(character, size):
    for i in range(size):
        row = ""
    for j in range(size):
        index = (i + j) % len(character)
        row += character[index]
    print(row)"""

import math
math.function()

"""vdu lap ra callfunction.py
dung from bai1 import test_function -> chuyen tu file nay sang file kia"""

"""def test_function(a):
    return a"""

"""if __name__ == "__main__":
print(test_function(10))""" # ko nen de test code sau function, nen de sau if

"""import math_operations as a
print(a.addition(10, 5))"""

# . la folder hien tai, .. la folder truoc do

#lambda là từ khóa để định nghĩa biểu thức lambda.
#a là đối số của hàm.
#a if a > 0 else 0 là biểu thức điều kiện. Nếu điều kiện a > 0 đúng, biểu thức trả về giá trị a, ngược lại trả về giá trị 0.
triple = lambda a: a if a > 0 else 0

#return 2 function
"""def area(a, b):
    return a *b, a+b

CV, DTICH= area(3,4)"""
# phai lay so bien = so kqua return
#enamurate lay index tung phan tu trong 1 chuoi
#vdu enamurate["a", "b", "c"] -> chuyen thanh [(0, "a"), (1, "b"), (2, "c")]
#vdu: for i, j in enamurate["a", "b", "c"]

a= ["a", "b" ,"c"]
b= ["c", "d", "e"]

#zip() là một hàm được sử dụng để kết hợp các phần tử từ các chuỗi (iterable) khác nhau thành các cặp tương ứng.
for i, j in zip(a, b):
    print(i + j)

#kqua se la [("a", "c"), ("b", "d"), ("c", "e")]
"""def area(a, b, c , *d)
def area(1, 2, 3, 4, 5) -> 4 va 5 la var
-> *d la nhieu hon 1 bien, coi la tuple """

#*lists cho phép chúng ta truyền vào bất kỳ số lượng danh sách (lists) nào vào hàm này
def interleave(*lists):
    results =[]

#kết hợp các phần tử từ các danh sách truyền vào hàm thành các tuple tương ứng.
    for i in zip(*lists):
        results.extend(i)
    return results

"""d.keys() -> [key1, key2]

d.values() -> [value1, value2]

d.items() -> [(key1, value1), (key2, value2)]"""

import math
pi= 3.14
print(round(pi)) #làm tròn đến số nguyên gần nhất
print(math.ceil(pi)) #làm tròn lên
print(math.floor(pi)) # làm tròn xuống
print(abs(pi)) #giá trị tuyệt đối
print(pow(pi, 2)) #pi luỹ thừa 2
print(math.sqrt(pi)) # tính căn bậc 2

"""x= 1
y= 2
z= 3
print(max(x, y, z))
print(min(x, y , z))"""

#WHILE
name =""
while len(name) == 0:
    name= input("enter your name:")

print("Hello" + name)

#NESTED LOOP
rows= int(input("how many rows?:"))
columns =int(input("how many columns?:"))
symbol= input("enter a symbol to use:")
for i in range(rows): #i dai dien cho hang
    for j in range(columns): #j dai dien cho cot
        print(symbol, end = "")
    print() #cach xuong dong

#LIST
food= ["cake", "hamburger", "soda"]
food[0]= "sushi" #thay vi tri 0 bang gtri sushi
food.append("ice cream") #them ice cream vao cuoi list
food.remove("hotdog") # xoa ten khoi list
food.pop() # xoa ten cuoi list
food.insert(0, "abc") # doi vi tri 0 thanh abc
food.sort() # xep list theo thu tu bang chu cai
food.clear() # xoa het gtri trong list

for i in food:
    print(i)

#2D LISTS = a list of lists

"""drinks= ["coffee", "soda", "tea"]
dinner= ["pizza", "hamburger", "hotdog"]
dessert = ["cake", "ice cream"]
food = [drinks, dinner, dessert]
print(food[1][2]) -> tim ra o list index vi tri 1 va gtri index 2 trong list do"""

#TUPLE
student= ("bro", 21, "male")
print(student.count("bro")) # dem so luong cua tu
print(student.index("male")) # tim index cua tu can tim
for x in student:
    print(x)
if "bro" in student:
    print("bro is here")

#SET -> no duplicate values
utensils= {"fork", "spoon", "knife"}
dishes ={"bowl", "plate", "cup", "knife"}
utensils.add("napkin") # them gtri napkin vao set utensils
utensils.remove("fork") # xoa gtri fork ra khoi set utensils
utensils.clear() # xoa het gtri trong utensils
dishes.update(utensils) # them gtri cua utensils vao dishes
dinner_table= utensils.union(dishes) # ket hop gtri dishes va utensils va dat ten la dinner_table

for x in dinner_table:
    print(x)

#DICTIONARY key:value
capitals = { "USA": " Washington DC",
            "India": "New Delhi",
            "China": "Beijing",
            "Russia": "Moscow"}

capitals.update({"Germany": "Berlin"}) # them gtri nay vao dict
capitals.pop("China") # xoa gtri china khoi list
capitals.clear() # xoa het cac gtri trong list
print(capitals["Germany"]) # tim value tu key
print(capitals.get("Germany")) # tim value tu key
print(capitals.keys()) # tim tat ca keys cua dict
print(capitals.values()) # tim cac value cua dict
print(capitals.items()) # in ra tat ca key va value
for key, value in capitals.items():
    print(key, value)

#INDEX OPERATOR
name="bro code"
if(name[0].islower()):
    name = name.capitalize()

first_name= name[:3].upper()
last_name= name[4:].lower()
last_character= name[-2]
print(first_name)
print(last_name)
print(last_character)
print(name)

"""def reserve_dictionary(dictionary):
    reverse_dict = {}
    for eng_word, fin_words in dictionary.items():
        for fin_word in fin_words:
            if fin_word in reverse_dict:
                reverse_dict[fin_word].append(eng_word)
            else:
                reverse_dict[fin_word] = [eng_word]
    return reverse_dict"""

"""def reserve_dictionary(dictionary):
reverse_dict = {}
for eng_word, fin_words in dictionary.items():
    for fin_word in fin_words:
        if fin_word in reverse_dict:
            reverse_dict[fin_word].append(eng_word)
        else:
            reverse_dict[fin_word] = [eng_word]

return reverse_dict"""

"""def find_matching(l: list, s:str) -> list:
        result= []
        for w in l:
            for s in w:
                result.append(w.index(s))"""

#(lambda number1, number2: number1 * number2)(5, 3) -> chay lambda khong can dat ten
numbers = [1, 2, 3, 4, 5]
def square(x):
    return x**2

squared_numbers = map(square, numbers)
print(list(squared_numbers))

"""l =[1, 2, 3, 4]
result= []
for i in l:
    result.append(i ** 2)
result

-> doi thanh tuong tu:
l= [1, 2, 3 ,4]
result= [i ** 2 for i in l]"""

"""def find_matching(l:list, s:str) -> list:
    result= [w.index(s) if s in w else -1 for w in l]
    return result

print(find_matching(l, s))"""

"""def reverse_dictionary(d: dict) -> dict:
result= {}
for key, values in d.items():
    for value in values:
        if value not in result:
            result[value] = [key]
        else:
            result[value].append(key)

return result"""

"""result= ""
for num in range(1999, 3201):
if num % 7 == 0 and num % 5 != 0:
    result += str(num) + ","

print(result.strip(","))
print(result)"""

"""import math
C = 50
H = 30
input_sequence = input("nhap cac gia tri d: ")
values = input_sequence.split(",")

result = []
for value in values:
    D = int(value)
    Q = math.sqrt((2 * C * D) / H)
    result.append(str(round(Q)))

output = ", ".join(result)
print(output)"""

"""\s -> cach
\t -> T
\n -> xuong dong"""

"""result= []
for i in range(2000, 3201):
    if i % 7 ==0 and i % 5 !=0:
        result.append(str(i))

",".join[str(i) for i in range(2000, 3201) if (i % 7 == 0) and (i % 5 != 0)]"""

"""def make_3sg_form(string):
if string.endswith("y"):
    return string[:-1] + "ies"
elif string.endswith(("o", "ch", "s", "sh", "x", "z")):
    return string + "es"
else:
    return string + "s"""

# **KWARGS
"""def hello(**kwargs):
    print("hello" + kwargs["first"]+ " " + kwargs["last"])
    print("hello", end= "")
    for key, value in kwargs.items():
        print(value, end = "")

hello(first="bro", middle= "dude", last ="code")"""

#random number

"""import random

x= random.randint(1, 6)
y= random.random()
mylist= ["rock", "paper", "scissors"]
z= random.choice(mylist)
cards= [1,2, 3, "j", "q", "k"]
random.shuffle(cards)
print(cards)"""

#EXCEPTION

"""try:
numerator =int(input("enter a number to divide: "))
denominator= int(input("enter a number to divide by: "))
result= numerator / denominator
except ZeroDivisionError:
    PRINT("U CANT DIVIDE by zero")
except ValueError:
    print("enter only number only")
except Exception:
    print("sth went wrong")
else:
    print(result)"""

#lambda parameters:expression

"""def double(x):
    return x * 2
print(double(5))
double= lambda x:x*2
multiply= lambda x, y:x * y
add= lambda x, y, z: x *y * z
full_name= lambda first_name, last_name: first_name+ " " + last_name
age_check= lambda age:True if age >= 18 else False
print(age_check(18))"""

#Sort() method = used with lists

#sort() function =used with iterables
"""students = ["squidward", "sandy", "patrick", "spongebob", "mr.krabs"]
students.sort() #sap xep theo thu tu chu cai
students.sort(reverse=True)
sorted_students= sorted(students, reverse=True)
for i in sorted_students:
    print(i)
for i in students:
    print(i)"""

"""students=[("squidward", "F", 60),
            ("sandy", "A", 33),
            ("patrick", "D", 36),
            ("spongebob", "B", 20),
            ("mr.krabs", "C", 78)]
age= lambda ages:ages[2]
students.sort(key= age, reverse=True)
sorted_students= sorted(students, key= age)
for i in sorted_students:
    print(i)"""

#map(function, iterable)
"""store= [("shirt", 20.00),
            ("pants", 25.00),
            ("jacket", 50.00),
            ("socks", 10.00)]

to_euros= lambda data: (data[0], data[1] * 0.82)
to_dollars= lambda data: (data[0], data[1]/0.82)
store_dollars =list(map(to_dollars, store))
for i in store_dollars:
    print(i)"""

#filter(function, iterable)
"""friends= [("rachel", 19),
            ("monica", 18),
            ("phoebe", 17),
            ("joey", 16),
            ("chandler", 21),
            ("rosa", 20)]

age= lambda data:data[1] >= 10
drinking_buddies= list(filter(age, friends))
for i in drinking_buddies:
    print(i)"""

#reduce(function, iterable)
"""import functools
letters= ["h", "e", "l", "l", "l", "o"]
word= functools.reduce(lambda x, y, : x+y, letters)
print(word)
factorial= [20, 30]
result= functools.reduce(lambda x, y, :x+y, factorial)
print(result)"""

#dictionary comprehension

#dictionary = {key: expression for {key, value} in iterable}
"""cities_in_F = {"New York": 32, "Boston":75, "Los Angeles": 100, "Chicago": 50}
cities_in_C = {key: round((value-32)*(5/9)) for (key, value) in cities_in_F.items()}
print(cities_in_C)"""

#-----------------------------------------------------------------------------------
"""import numpy as np
np.array([1, 2, 3])"""

#np.array([1, 2, 3]).dtype -> check kieu du lieu

# np.array(data, dtype) data: list dtype: bool, float, int

#np.array([1, 2, 3], dtype= bool) -> chuyen kieu du lieu

#np.array([1, 2, 3], dtype= float).shape -> tra ve kieu du lieu

# 1D la 1 list , 2D la cac hang cac cot, 3D la cac 2D chong len nhau

#a.shape -> tra ve bao nhieu gia tri thi la bay nhieu chieu -> tra ve tong so phan tu trong chieu do

#np.array([1, 2, 3], dtype= bool).ndim -> tim so chieu

#np.array([1, 2, 3], dtype= bool).size ->

#np.zeros((5, 5), dtype= bool)

#np.ones((5,5), dtype= bool)

#np.eye(5) -> tao ra 1 ma tran

#np.zeros(a.shape)

#np.zeros_like(array)

#np.ones(shape)

#np.eye(a: int) -> array: axa

#np.full((2, 3), fill_value= 1) -> fillvalue: du lieu truyen vao, (2, 3) la shape

#np.random.random(3,4)

"""import numpy as np
a= np.random.random(shape) -> return random float"""

#np.random.randint()

#np.arange()

#np.linspace(start, stop, so phan tu) -> tim ra cac so cach deu nhau theo so phan tu ban nhap

#np.random.seed(10) -> tao ra cac so nguyen giong nhau

#a[1:3, 1:3]

#a[3, :]

#a[hang, cot] a[[0, 2, 3], 3] -> fancy index

# a[mask] -> a.shape= mask.shape

# a[a % 2 == 0] -> cac phan tu la true

#a.reshape((4,2))

#a.reshape((4, -1)) -> cai nao la -1 thi tu tinh

#np.concatenate((a,b,c), axis= ) axis =0 ->hang, axis=1 -> cot, axis= 2 -> 2D

"""import numpy as np

# Create a 1D array of 10 elements
array_1d = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Replace even numbers with -1
array_1d[array_1d % 2 == 0] = -1
print(array_1d)"""

"""import numpy as np
def diamond(side_length):
    matrix = np.eye(side_length, dtype= int)
    matrix_2 = np.flip(matrix[:, :2], axis = 0)
    a= np.concatenate((matrix_2, matrix), axis= 1)
    b= np.flip(a, axis= 0)
    c= np.concatenate((a, b[1:]), axis =0)
    return c

print(diamond(3))"""

"""import numpy as np
array = np.random.randint(1, 11, size= (3, 4))
print(array)"""

"""import numpy as np
array= np.array([2, 4, 6, 8, 10])
array_2 = array * 3 - 5
print(array_2)"""

"""import numpy as np
array= np.arange(10)
array[array % 2 == 0] = -1
print(array)"""

"""import numpy as np
array= np.array([11,22,33,44,55,66,77,88,99])
print(array[2:7:2])"""

"""import numpy as np
array =np.arange(1, 21).reshape(4, 5)
subarray = array[-2:, :3]
print(subarray)"""

"""import numpy as np
array= np.array([[1,2,3],
                [3, 4, 5],
                [4, 5, 6]])
num_column = array.shape[1]
array[:, [0, num_column-1]] = array[:, [num_column-1, 0]]
print(array)"""

'''import numpy as np
array = np.arange(1, 21).reshape(4, 5)
row_sums= np.sum(array, axis=1 )
print(row_sums)'''

"""import numpy as np
array= np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])
flatten_away = array.flatten()
reshape_away= flatten_away.reshape(2, 6)
print(reshape_away)"""

"""import numpy as np
a = np.array([1,2,3])
b= np.array([4,5,6])
horizontal_stack= np.hstack((a.reshape(-1, 1), b.reshape(-1,1)))
vertical_stack= np.vstack((a,b))
print(horizontal_stack)
print(vertical_stack)"""

"""import numpy as np
a= np.random.randint(-100, 100, (4,5))
print(a)
print(f"minimum:{a.min()}, maximum:{a.max()}")
print(f"sum:{a.sum()}")
print(f"mean: {a.mean()}, standard deviation:{a.std()}")"""

"""import numpy as np
np.random.seed(9)
b= np.random.randint(0, 10, (3,4))
print(b)
print("column sums:", b.sum(axis= 0))
print("row sums:", b.sum(axis= 1))"""

"""import numpy as np
a = np.arange(1, 10)
print(a)
a.resize((4,3))
print(a)"""

"""import numpy as np
def multiplication_table(n):
    row= np.arange(0, n)
    column = np.arange(0, n).reshape(-1, 1)
    table = row * column
    return table

print(multiplication_table(4))"""

"""import numpy as np
a = np.random.random((2,4))
print((a>0.5).any())"""

"""import numpy as np
a = np.random.random((2,4))
print((a>0.5).sum())"""

#BROADCASTING

#chi lam duoc khi so chieu bang 1

"""np.array zeros(shape) -> zeroslike(array)
constant ones(shape) -> oneslike
np.random.random full(shape) -> fulllike
np.random.randint eye(int)
info(): a.shape
a.dim
n.dtype
a.size
index, slice, masking
reshape
a.size= b.size
max, min, sum, mean, std, median
max(.., axis = 0 / 1)
~ la phu dinh"""

"""import numpy as np
df= [np.random.randint(1, 50) for i in range(20)]
print(df)

#flatten using reshape(1, -1)
array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_array = [num for num in array if num % 2 == 0]
print(even_array)

import numpy as np
rgb_images = np.random.randint(0, 256, size= (3, 4, 4, 3))
flattened_images = np.reshape(rgb_images, (rgb_images.shape[0], -1))
stacked_image = np.hstack(flattened_images)
print(stacked_image)"""

"""import numpy as np
image = np.random.random((5,5))
mask = np.zeros_like(image, dtype=bool)
print(image)

sub_array = image[1:-1, 1:-1]
argmax_indices = np.argmax(sub_array, axis=None)
max_row, max_col = np.unravel_index(argmax_indices, sub_array.shape)
max_row += 1
max_col += 1
mask[max_row, max_col] = True
print(mask)"""

"""import numpy as np
random_array= np.random.randint(1, 50, size = 20)
print(random_array)"""

"""import numpy as np
def division(array_2d, array_1d):
    result = array_2d / array_1d[:, np.newaxis]
    return result
array_2d= np.random.random((3, 3))
array_1d = np.random.random((1,3))
print(array_2d)
print(array_1d)
result =division(array_2d, array_1d)
print(result)"""

"""import numpy as np
array_3d = np.random.random((5,5,5))
array_2d = array_3d.reshape((25,5))
print(array_2d)"""

"""import numpy as np
scores = np.array([[80, 90, 85],
                    [60, 70, 75],
                    [90, 95, 92],
                    [70, 80, 65]])
threshold = 80

import numpy as np
array_2d = np.array([[1,2,3],
                    [4,5,6],
                    [7,8,9]])
mean_value = np.mean(array_2d)
mask = array_2d > mean_value
print(mask)"""

"""import numpy as np
temperature = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
threshold = 30
mask = temperature >= threshold
print(mask)"""

"""import numpy as np
array_2d = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 15])
mask3 = (array_2d % 3 == 0)
mask5 = (array_2d % 5 == 0)
maskbine= mask3 & mask5
abc = array_2d[maskbine]
print(abc)
"""

#df[""].unique().size

#df[""].unique()

#df[""].value_counts()

#df[""].fillna(0) , dropna(axis= 1/ 0)

#df.isnull().sum()

#df.to_csv('suicide_dataset_cleaned.csv', index=False)

#df = df.dropna(subset=['VEHICLECLASS'])

#df.columns

#.groupby(["sex", "country"])

#df["country"].value_counts()["Albania"]

#df.groupby(["country", "sex"]).get_group(("ABC", "FEMALE")).max()

#agg -> apply func to DF

#df.agg(["mean", "sum", "max"])

#df.agg({"..".average})

#string

#isnumber, contain, islower

#df["country"].str.contains("Alba")

#df["country"].str.replace("Alba", "XYZ") -> .str : bien thanh string

#pd.to_datetime(df["Year"])

#df[df["org_date"] > "2012-10-01"]

#df[(df["org_date"] > "2012-10-01") & (df["org_date"]>"2012-07-01

#df.grouper(pd.grouper(key= "org_date", freq= "M")).value_counts()

#--------------------------------------------------

#plt.bar

#plt.hbar

"""import pandas as pd
abc= pd.read_csv(r"c:\Users\thengoc\Downloads\weather.csv")
avg_temp= abc.groupby(["m"])["Air temperature (degC)"].mean()
month = avg_temp.index
temp_mean = avg_temp.values
import matplotlib.pyplot as plt
plt.plot(month, avg_temp, color = "red")
plt.title("Weather")
plt.xlabel("Month")
plt.ylabel("Air temperature (degC)")
plt.show()"""

import pandas as pd
abc= pd.read_csv(r"c:\Users\thengoc\Downloads\weather.csv")
avg_temp= abc.groupby(["m"])["Air temperature (degC)"].mean()
avg_snowdepth = abc.groupby(["m"])["Snow depth (cm)"].mean()
avg_pre= abc.groupby(["m"])["Precipitation amount (mm)"].mean()
month = avg_temp.index
temp_mean = avg_temp.values
import matplotlib.pyplot as plt
plt.plot(month, avg_temp, color = "red", label= "temp")
plt.plot(month, avg_snowdepth, color= "yellow", label= "snow")
plt.plot(month, avg_pre, color = "orange", label= "pre")
plt.title("Weather")
plt.xlabel("Month")
plt.legend()
plt.show()