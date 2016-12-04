from glob import glob
from tkinter import Canvas
import tkinter as tk
from PIL import ImageTk, Image
import pickle


class ImageRectangleFrame(tk.Frame):
    """
    A Frame to draw rectangles on an image.
    """
    def __init__(self, parent, width, height, image_path, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)

        self.rectangles_coordinates = {}
        self.image = None
        self.canvas = self._create_canvas(width, height)
        self.set_image(image_path)

    def _create_canvas(self, width, height):
        """
        Create a canvas
        :param width: Width of the canvas
        :param height: Height of the Canvas
        :return: Canvas
        """
        canvas = Canvas(self, width=width, height=height)
        canvas.pack()
        canvas.bind('<ButtonPress-1>', self._key_button_1_press)
        canvas.bind('<B1-Motion>', self._key_button_1_motion)
        canvas.bind('<ButtonRelease-1>', self._key_button_1_release)

        return canvas

    def _key_button_1_press(self, event):
        """
        Create a rectangle
        :param event: The mouse event
        """
        self.current_rectangle = self.canvas.create_rectangle((event.x, event.y, event.x, event.y))

    def _key_button_1_motion(self, event):
        """
        Modify the rectangle as the mouse drags the cursor
        :param event: The mouse event
        """
        current_x1, current_y1, _, _ = self.canvas.coords(self.current_rectangle)
        self.canvas.coords(self.current_rectangle, current_x1, current_y1, event.x, event.y)

    def _key_button_1_release(self, _):
        """
        Save the rectangle coordinates
        """
        self.rectangles_coordinates[self.current_rectangle] = self.canvas.coords(self.current_rectangle)
        self.current_rectangle = None

    def set_image(self, image_path):
        """
        Set the image for the canvas
        :param image_path: Path of the image
        """
        # Save image to the class, so it won't be garbage collected
        self.image = ImageTk.PhotoImage(Image.open(image_path))
        # Put the image in the center of the canvas
        self.canvas.create_image(0, 0, anchor='nw', image=self.image)

    def clear(self):
        """
        Clear the canvas
        """
        # Delete Canvas Items
        for rectangle_id in self.rectangles_coordinates.keys():
            self.canvas.delete(rectangle_id)
        self.canvas.delete(self.image)

        self.image = None
        self.rectangles_coordinates = {}


class MainApplication(tk.Frame):
    """
    Manually label all the fish locations
    """
    def __init__(self, parent, width, height, image_paths, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)

        assert len(image_paths) > 0
        self.image_paths = image_paths
        self.all_fish_coordinates = {}
        self.current_image_id = 0

        self.parent = parent
        self.parent.bind('<BackSpace>', self._key_backspace)
        self.parent.bind('<space>', self._key_space)

        self.image_classification = ImageRectangleFrame(self, width, height, self.image_paths[self.current_image_id])
        self.image_classification.pack()

    def _key_backspace(self, _):
        """
        Clear the image
        """
        self.image_classification.clear()
        self.image_classification.set_image(self.image_paths[self.current_image_id])

    def _key_space(self, _):
        """
        Save the found fish and go to the next image
        """
        self.all_fish_coordinates[self.image_paths[self.current_image_id]] =\
            self.image_classification.rectangles_coordinates
        self.image_classification.clear()
        self.current_image_id += 1

        # Check if it's the last image
        if self.current_image_id < len(self.image_paths):
            self.image_classification.set_image(self.image_paths[self.current_image_id])
        else:
            self.quit()


def run():
    """
    Manually label all the fish locations and save it to file
    """
    train_images_path = './data/train/*/*.jpg'
    output_filename = 'fish_coordinates.p'
    image_size = (1280, 720)
    tk_root = tk.Tk()

    app = MainApplication(tk_root, *image_size, list(glob(train_images_path)))
    app.pack(side='bottom', fill='both', expand=True)
    tk_root.mainloop()

    with open(output_filename, 'w+b') as outfile:
        pickle.dump(app.all_fish_coordinates, outfile)

if __name__ == '__main__':
    run()
