import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import Qt, QUrl, QTimer
from pathlib import Path
import os

import random


class CustomButton(QPushButton):

    style = """
    QPushButton {
    border: 1px solid #6EB1FF;
    background-color: #00072C;
    color: #C8F4FF;
    padding: 12px;
    margin: 6px 18px;
    border-radius: 16px;
    width: 350px;
    font-size: 16pt
    }
    QPushButton:hover {
    background-color: #001339
    }
"""

    def __init__(self, text):
        super().__init__(text)

        self.setStyleSheet(self.style)
        self.setCursor(Qt.PointingHandCursor)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PyQt5 Base Template")
        self.setGeometry(100, 100, 1200, 800)

        self.moods = ["Angry", "Sad", "Happy", "Relaxed"]
        self.mood_of_music: str = None
        self.correct_answers = 0
        self. played_musics = 0
        self.widgets = []
        
        self.start_new_song()
        self.init_ui()
        

    def init_ui(self):
        # Central widget (gpt gave this dunno exactly why i need it)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        self.setStyleSheet("""
        background-color: #00001E; 
""")
        # Layout
        layout = QVBoxLayout()

        self.score_label = QLabel("Score: 0/0")
        self.score_label.setStyleSheet("""
        font-size: 14pt;
        color: #C8F4FF;
        margin: 8px;
""")
        self.widgets.append(self.score_label)
        
        label = QLabel("What is the mood of played song?")
        
        self.widgets.append(label)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("""
        font-size: 22pt;
        color: #C8F4FF;
        margin-bottom: 16px;
""")
        
        # create buttons
        buttonLayout = QVBoxLayout()
        buttonLayout.addStretch()
        for i in range(2):
            rowLayout = QHBoxLayout()
            rowLayout.addStretch()
            for j in range(2):
                button = CustomButton(self.moods[i * 2 + j])
                button.clicked.connect(self.on_button_click)
                self.widgets.append(button)
                rowLayout.addWidget(button)
            rowLayout.addStretch()
            buttonLayout.addLayout(rowLayout)
        buttonLayout.addStretch()

        # Add widgets to layout
        layout.addWidget(self.score_label)
        layout.addStretch(8)
        layout.addWidget(label)
        layout.addStretch(4)
        layout.addLayout(buttonLayout)
        layout.addStretch(8)

        central_widget.setLayout(layout)

    def on_button_click(self):
        self.player.stop()
        
        # answer checking logic
        text_on_button = self.sender().text()
        is_correct = self.check_answer(text_on_button)
        print(is_correct)
        if is_correct:
            self.correct_answers += 1
        print(text_on_button, self.mood_of_music)
        
        # make the widgets go and come like something is happening (but there is nothing happening)
        self.make_the_widgets_pufff()
        QTimer.singleShot(100, self.rebirth_of_the_wdgets)

        print(self.played_musics, self.correct_answers)

        # update the score duh
        self.score_label.setText(f"Score: {self.correct_answers}/{self.played_musics}")

        self.start_new_song()
        

    def check_answer(self, response: str):
        return response == self.mood_of_music
    
    def start_new_song(self):
        data_folder_path = os.path.abspath("data")
        music_path = Path(data_folder_path) / self.pick_a_song()

        self.player = QMediaPlayer()
        self.player.setMedia(
            QMediaContent(QUrl.fromLocalFile(str(music_path)))
        )
        self.player.play()

    def pick_a_song(self):
        self.played_musics += 1
        self.mood_of_music = self.moods[random.randint(0, 3)]
        song_number= random.randint(1, 199)
        song_path = Path(self.mood_of_music.lower()) / f"{self.mood_of_music.lower()}{song_number}.mp3"
        
        return song_path
    
    # make the widgets pufff. Something to audience love (audience is me)
    def make_the_widgets_pufff(self):
        for widget in self.widgets:
            widget.setVisible(False)

    # rebirth of the widgets (don't wait 3 days)
    def rebirth_of_the_wdgets(self):
        for widget in self.widgets:
             widget.setVisible(True)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
