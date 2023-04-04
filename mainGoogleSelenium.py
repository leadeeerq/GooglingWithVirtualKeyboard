import cv2
from cvzone.HandTrackingModule import HandDetector
from time import sleep, time
import numpy as np
from pynput.keyboard import Controller
from selenium import webdriver

BUTTON_COLOR = (255, 255, 255)
FONT_COLOR = (0, 0, 0)
START_POS = (50, 50)
SIZE_OF_BUTTON = (85, 85)

final_text = ""

# Capturing the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)  # Height

# Hand detector object
detector = HandDetector(detectionCon=0.8)

# Keyboard object
keyboard = Controller()


# Function for drawing the keyboard on the screen
def draw_all(img, list_of_buttons):
    img_new = np.zeros_like(img, np.uint8)
    for button in list_of_buttons:
        cv2.rectangle(img_new, button.pos, (button.pos[0] + button.size[0], button.pos[1] + button.size[1]),
                      BUTTON_COLOR, cv2.FILLED)
        cv2.putText(img_new, button.text, (button.pos[0] + 15, button.pos[1] + 70), cv2.FONT_HERSHEY_PLAIN, 5,
                    FONT_COLOR, 5)
    # Changing the transparency of the keyboard
    overlay = img.copy()
    alpha = 0.5
    mask = img_new.astype(bool)
    overlay[mask] = cv2.addWeighted(img, alpha, img_new, 1 - alpha, 0)[mask]
    return overlay


# Function to submit the text to google search
def googler(final_text):
    base_url = 'https://google.com/search?q='
    url = base_url + str(final_text)
    driver = webdriver.Chrome()
    driver.get(url)
    driver.maximize_window()
    driver.implicitly_wait(3)
    # Find the policy acceptance button and click it
    driver.find_element_by_xpath('//*[@id="L2AGLb"]/div').click()
    driver.implicitly_wait(3)
    # Wait for 5 sec
    sleep(5)
    # And close the browser
    driver.quit()


# Class of a single button
class Button():
    def __init__(self, pos, text, size=SIZE_OF_BUTTON):
        self.pos = pos
        self.text = text
        self.size = size


# First row: Q W E R T Y U I O P
first_row_letters = list('QWERTYUIOP')
# Second: A S D F G H J K L and backspace as "<"
second_row_letters = list('ASDFGHJKL<')
# Third: Z X C V B N M
third_row_letters = list('ZXCVBNM,./')

longest_row = max(first_row_letters, second_row_letters, third_row_letters, key=len)
max_row_length = len(longest_row)

# Creating a list of buttons positions in X axis
positions_x = [START_POS[0]]
for i in range(max_row_length):
    next_pos = positions_x[-1] + SIZE_OF_BUTTON[0] + 5
    positions_x.append(next_pos)

# Creating the buttons aligned in rows
first_row_buttons = []
for letter, pos in zip(first_row_letters, positions_x):
    first_row_buttons.append(Button((pos, START_POS[1]), letter))

second_row_buttons = []
for letter, pos in zip(second_row_letters, positions_x):
    second_row_buttons.append(Button((pos + 20, START_POS[1] + 90), letter))

third_row_buttons = []
for letter, pos in zip(third_row_letters, positions_x):
    third_row_buttons.append(Button((pos + 40, START_POS[1] + 180), letter))

list_of_buttons = first_row_buttons + second_row_buttons + third_row_buttons

# Space
list_of_buttons.append(
    Button(pos=(positions_x[2] + 40, START_POS[1] + 270), text=' ', size=(positions_x[2] + 215, START_POS[1] + 35)))

# Google button
list_of_buttons.append(
    Button(pos=(900, 550), text='Google', size=(330, 100)))

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)

    # List of hand landmarks
    lmList, bbInfo = detector.findPosition(img)

    img = draw_all(img, list_of_buttons)

    # Check whether a hand present
    if lmList:
        for button in list_of_buttons:
            x, y = button.pos
            w, h = button.size
            # Index fingertip is marked with number 8
            # Check if X and Y position of the fingertip is in the rectangle
            if (x < lmList[8][0] < x + w) and (y < lmList[8][1] < y + h):
                cv2.rectangle(img, button.pos, (button.pos[0] + button.size[0], button.pos[1] + button.size[1]),
                              (120, 120, 120), cv2.FILLED)
                cv2.putText(img, button.text, (button.pos[0] + 15, button.pos[1] + 70), cv2.FONT_HERSHEY_PLAIN, 5,
                            FONT_COLOR, 5)
                # Check the distance between index and thumb tip - small distance will mean a click
                l, _, _ = detector.findDistance(8, 4, img)
                # Check the length to set the threshold
                # print(l)
                if l < 55:
                    if button.text != 'Google':
                        keyboard.press(button.text)
                        # sleep(0.1)
                        # keyboard.release(button.text)
                        cv2.rectangle(img, button.pos, (button.pos[0] + button.size[0], button.pos[1] + button.size[1]),
                                      (0, 0, 0), cv2.FILLED)
                        cv2.putText(img, button.text, (button.pos[0] + 15, button.pos[1] + 70), cv2.FONT_HERSHEY_PLAIN,
                                    5,
                                    (255, 255, 255), 5)
                        if button.text == "<" and len(final_text) > 0:
                            final_text = final_text[:-1]
                        elif button.text == "<" and len(final_text) == 0:
                            final_text = final_text
                        elif len(final_text) < 14:
                            final_text += button.text
                        sleep(0.3)
                    else:
                        print('Google it.')
                        googler(final_text)

    cv2.rectangle(img, (50, 550), (850, 650), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, final_text, (60, 625), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
