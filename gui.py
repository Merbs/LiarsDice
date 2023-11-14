import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QGridLayout, QPushButton
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QPixmap

from player import RandomPlayer, HumanPlayer
from game import Game
from player import Player
import numpy as np
import time

import functools

class HumanGUIPlayer(Player):
    
    def __init__(self,name=None,gui=None):
        super().__init__(name)
        self.gui = gui
    
    def get_move_scores(self,board: np.array) -> np.array:
        n_cols = board.shape[2]
        invalid_input = True
        while invalid_input:
            new_input = input(f"Select column to drop token into [0-{n_cols-1}]\n")
            try:
                slot_to_drop = int(new_input)
            except ValueError:
                continue
            
            if 0 <= slot_to_drop and slot_to_drop < n_cols:
                invalid_input = False
                
        scores = np.zeros(n_cols)
        scores[slot_to_drop] = 1
        return scores
    
    def get_move_scores_gui(self,board: np.array) -> np.array:
        n_cols = board.shape[2]
        
        # Waiting for button press
        slot_to_drop = None
        while slot_to_drop is None:
            #slot_to_drop = gui.last_button_pressed() # ??
            return np.array([1,0,0,0,0,0,0])
            time.sleep(0.10)
                
        scores = np.zeros(n_cols)
        scores[slot_to_drop] = 1
        return scores

def window(game: Game):
    app = QApplication(sys.argv)
    win = QWidget()
    grid = QGridLayout()
    	
    nrows = 6
    ncols = 7
    
    
    empty_pixmap = QPixmap('empty.png')
    empty_pixmap = empty_pixmap.scaledToWidth(100)
    red_pixmap = QPixmap('empty.png')
    red_pixmap = red_pixmap.scaledToWidth(100)
    blue_pixmap = QPixmap('empty.png')
    blue_pixmap = blue_pixmap.scaledToWidth(100)
    
    label_grid = []
    
    for i in range(nrows):
        label_row = []
        for j in range(ncols):
           
            label = QLabel()
            label.setPixmap(empty_pixmap)
            grid.addWidget(label,i,j)
            label_row.append(label)
           
        label_grid.append(label_row)
           
    @pyqtSlot()
    def change_picture(j):
        pixmap = QPixmap('red.png')
        pixmap = pixmap.scaledToWidth(100)
        label_grid[-1][j].setPixmap(pixmap)
           
    for j in range(ncols):
        drop_button = QPushButton("Drop")
        grid.addWidget(drop_button,nrows,j)
        drop_button.clicked.connect(functools.partial(change_picture, j))
        
    # Start game button
    @pyqtSlot()
    def on_click():
        for turn in game.play_game(show_board_each_move=True,verbose=True):
            for i in range(nrows):
                for j in range(ncols):
                    if game.board[0][i][j] == 1:
                        label_grid[-1][j].setPixmap(red_pixmap)
                    elif game.board[1][i][j] == 1:
                        label_grid[-1][j].setPixmap(blue_pixmap)

    start_button = QPushButton("Start Game")
    start_button.clicked.connect(on_click)
    grid.addWidget(start_button,nrows+1,0)
        
    			
    win.setLayout(grid)
    win.setWindowTitle("PyQt Grid Example")
    win.setGeometry(50,50,200,200)
    win.show()
    
    #game.play_game(show_board_each_move=True)
    
    sys.exit(app.exec_())

def main():
    g = Game()
    human = HumanGUIPlayer(name="Human")
    g.players = [human,RandomPlayer(name="Random Bot")]
    

    window(g)
    
    

if __name__ == '__main__':
   main()