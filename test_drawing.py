import tkinter as tk
from PIL import Image, ImageDraw
import sys

class DrawingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Draw Character for EMNIST")
        
        # Create canvas
        self.canvas_size = 280  # 10x the final size for better drawing
        self.canvas = tk.Canvas(master, width=self.canvas_size, 
                                height=self.canvas_size, bg='white')
        self.canvas.pack()
        
        # Setup drawing
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None
        self.pen_width = 15
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.start_draw)
        self.canvas.bind('<B1-Motion>', self.draw_line)
        self.canvas.bind('<ButtonRelease-1>', self.stop_draw)
        
        # Instructions
        label = tk.Label(master, text="Draw a character (digit or letter)")
        label.pack()
        
        # Buttons
        button_frame = tk.Frame(master)
        button_frame.pack()
        
        save_btn = tk.Button(button_frame, text="Save", command=self.save_image)
        save_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = tk.Button(button_frame, text="Clear", command=self.clear_canvas)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        quit_btn = tk.Button(button_frame, text="Quit", command=master.quit)
        quit_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_count = 0
        
    def start_draw(self, event):
        self.last_x = event.x
        self.last_y = event.y
        
    def draw_line(self, event):
        if self.last_x and self.last_y:
            # Draw on canvas
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                   width=self.pen_width, fill='black',
                                   capstyle=tk.ROUND, smooth=tk.TRUE)
            # Draw on PIL image
            self.draw.line([self.last_x, self.last_y, event.x, event.y],
                         fill='black', width=self.pen_width)
            self.last_x = event.x
            self.last_y = event.y
            
    def stop_draw(self, event):
        self.last_x = None
        self.last_y = None
        
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 'white')
        self.draw = ImageDraw.Draw(self.image)
        
    def save_image(self):
        # Resize to 28x28 for EMNIST
        small_image = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        filename = f'test_char_{self.save_count}.png'
        small_image.save(filename)
        print(f"Saved as {filename}")
        
        # Also save the full-size version for reference
        full_filename = f'test_char_{self.save_count}_full.png'
        self.image.save(full_filename)
        print(f"Full size saved as {full_filename}")
        
        self.save_count += 1
        self.clear_canvas()

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    print("Draw a character and click 'Save' to create test images")
    print("Images will be saved as test_char_N.png")
    root.mainloop()