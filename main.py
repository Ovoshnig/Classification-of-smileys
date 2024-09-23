import tkinter as tk
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Создать окно tkinter
window = tk.Tk()
window.title("Распознавание смайликов")

# Размер пикселя по умолчанию
pixel_size = 20

# Флаг для выбора режима
is_training = True

# Структуры для хранения данных о смайликах
smiley_data = []
smiley_arr = np.zeros((8, 8), dtype=np.uint8)

smiley_label = []

# Выбор режима обучения или тестирования
train_label = tk.Label(window, text="Обучение", bg='lightgreen')
train_label.grid(column=0, row=0)

test_label = tk.Label(window, text="Тестирование", bg='lightyellow')
test_label.grid(column=2, row=0)

# Размеры окна 8x8 пикселей
canvas_width = 8 * pixel_size
canvas_height = 8 * pixel_size

# Флаг, который показывает, рисует ли пользователь
drawing = False

# Создать поле для рисования
canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg="white")
canvas.grid(column=1, row=0, rowspan=6)

# Функция для начала рисования
def start_drawing(event):
    global drawing
    drawing = True

# Функция для завершения рисования
def stop_drawing(event):
    global drawing
    drawing = False

# Функция для рисования пикселя
def draw_pixel(event):
    if drawing:
        x, y = event.x, event.y
        # Ограничиваем рисование только в окне 8x8 пикселей
        if 0 <= x < canvas_width and 0 <= y < canvas_height:
            global smiley_arr
            
            # Рисование пикселя с выбранным размером
            x1, y1 = x // pixel_size * pixel_size, y // pixel_size * pixel_size
            x2, y2 = x1 + pixel_size, y1 + pixel_size
            canvas.create_rectangle(x1, y1, x2, y2, fill="black")
            smiley_arr[y // pixel_size, x // pixel_size] = 1

# Привязать события мыши к функциям
canvas.bind("<ButtonPress-1>", lambda event: (start_drawing(event), draw_pixel(event)))
canvas.bind("<ButtonRelease-1>", stop_drawing)
canvas.bind("<B1-Motion>", draw_pixel)

# Выбор типа смайлика
smiley_type_label = tk.Label(window, text="Выберите тип смайлика:")
smiley_type_label.grid(column=0, row=1)

smiley_type = tk.StringVar()
smiley_type.set("Веселый")
happy_radio = tk.Radiobutton(window, text="Веселый", variable=smiley_type, value="Веселый")
sad_radio = tk.Radiobutton(window, text="Грустный", variable=smiley_type, value="Грустный")
happy_radio.grid(column=0, row=2)
sad_radio.grid(column=0, row=3)

# Функция для сохранения данных смайлика
def save_smiley_data():
    canvas.delete('all')

    global smiley_arr
    smiley_data.append(smiley_arr)
    smiley_arr = np.zeros((8, 8), dtype=np.uint8)

    global smiley_label
    global smiley_type
    if smiley_type.get() == "Веселый":
        smiley_label.append(1)
    else:
        smiley_label.append(0)

# Кнопка для сохранения данных смайлика
save_button = tk.Button(window, text="Сохранить смайлик", command=save_smiley_data)
save_button.grid(column=0, row=4)

def fit():
    global smiley_data
    smiley_data = np.array(smiley_data)
    global smiley_label
    smiley_label = np.array(smiley_label)

    try:
        file_data = np.load("data.npy")
        smiley_data = np.concatenate((file_data, smiley_data), axis=0)
        file_label = np.load("label.npy")
        smiley_label = np.concatenate((file_label, smiley_label), axis=0)
    except:
        pass
    np.save("data.npy", smiley_data)
    np.save("label.npy", smiley_label)

    smiley_data = smiley_data.reshape(-1, 8, 8, 1)
    
    # Разделение данных на тренировочные и валидационные
    x_train, x_valid, y_train, y_valid = train_test_split(smiley_data, smiley_label, test_size=0.2, random_state=42)
    
    # Создание модели
    model = Sequential()

    # Первый сверточный слой
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(8, 8, 1)))
    model.add(MaxPooling2D((1, 1)))

    # Второй сверточный слой
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((1, 1)))

    # Преобразование данных перед подачей на полносвязный слой
    model.add(Flatten())

    # Полносвязные слои
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Dropout для предотвращения переобучения
    model.add(Dense(1, activation='sigmoid'))  # Выходной слой для бинарной классификации

    # Компиляция модели
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Обучение модели
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_valid, y_valid))

    model.save("model.keras")

    smiley_data = []
    smiley_label = []

# Кнопка для обучения модели
save_button = tk.Button(window, text="Обучить", command=fit)
save_button.grid(column=0, row=5)

def classify():
    global smiley_arr
    sa = smiley_arr.reshape(1, 8, 8, 1)

    # Загрузка модели из файла
    model = load_model("model.keras")

    # Предсказание грустный или веселый смайлик
    pred = int(100 * model.predict(sa)[0, 0])
    if pred >= 50:
        class_label.configure(text=f"Веселый на {pred}%")
    else:
        class_label.configure(text=f"Грустный на {100 - pred}%")

# Надпись и кнопка для классификации смайликов
classify_label = tk.Button(window, text="Классифицировать смайлик", command=classify)
classify_label.grid(column=2, row=1)

class_label = tk.Label(window, text="")
class_label.grid(column=2,row=2)

def clear():
    canvas.delete('all')

    global smiley_arr
    smiley_arr = np.zeros((8, 8), dtype=np.uint8)

# Кнопка для очистки холста
clear_button = tk.Button(window, text="Очистить холст", command=clear)
clear_button.grid(column=1, row=6)

# Запустить приложение
window.mainloop()
