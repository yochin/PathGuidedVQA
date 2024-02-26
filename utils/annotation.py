import os
import tkinter as tk
from PIL import Image, ImageTk
import xml.etree.ElementTree as ET

class ImageAnnotator:
    def __init__(self, image_folder, anno_folder):
        self.image_folder = image_folder
        self.anno_folder = anno_folder
        self.images = [file for file in os.listdir(image_folder) if file.lower().endswith(('png', 'jpg', 'jpeg'))]
        self.index = 0
        self.annotations = []

        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.load_image()
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_drag_release)  # 드래그 릴리스 이벤트 바인딩
        self.root.bind("<Key>", self.on_key_press)
        self.drag_start = None  # 드래그 시작점 초기화
        self.drag_end = None  # 드래그 끝점 초기화
        self.temp_rect = None  # 임시 바운딩 박스 객체 초기화
        self.points = []  # For curve points

    def load_image(self):
        self.canvas.delete("all")

        if self.index < len(self.images):
            img_path = os.path.join(self.image_folder, self.images[self.index])
            self.img = Image.open(img_path)
            self.tkimg = ImageTk.PhotoImage(self.img)

            # Resize window to fit the image
            self.root.geometry(f"{self.img.width}x{self.img.height}")

            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tkimg)
            self.canvas.config(width=self.img.width, height=self.img.height, scrollregion=self.canvas.bbox(tk.ALL))
        else:
            self.save_annotations()
            self.root.destroy()

    def on_click(self, event):
        # Normalize coordinates and save
        self.gp_x, self.gp_y = event.x / self.img.width, event.y / self.img.height
        print(f"Click at normalized position: ({self.gp_x}, {self.gp_y})")

        # Draw a small circle at the click position as a visual cue
        r = 5  # Radius of the circle
        self.canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, fill='red')

    def on_drag(self, event):
        # For bounding box, assume this starts the drag for one corner and ends at the opposite corner
        self.x1, self.y1 = event.x, event.y
        print(f"Drag start at: ({self.x1}, {self.y1})")

    def on_key_press(self, event):
        if event.char == "g":
            self.go_or_stop = "GO"
        elif event.char == "s":
            self.go_or_stop = "STOP"
        elif event.keysym == "Return":
            self.save_annotation()
            self.index += 1
            self.load_image()

    def save_annotation(self):
        # Assuming x2, y2 are set at the end of the drag; implement accordingly
        annotation = {
            "image": self.images[self.index],
            "gp_x": self.gp_x,
            "gp_y": self.gp_y,
            "bbox": [self.x1, self.y1, self.x2, self.y2],
            "go_or_stop": self.go_or_stop,
            "curve_points": self.points,
        }
        self.annotations.append(annotation)

    def save_annotations(self):
        for annotation in self.annotations:
            # Generate XML for each annotation
            root = ET.Element("annotation")
            ET.SubElement(root, "filename").text = annotation["image"]
            ET.SubElement(root, "gp_x").text = str(annotation["gp_x"])
            ET.SubElement(root, "gp_y").text = str(annotation["gp_y"])
            bbox = ET.SubElement(root, "bbox")
            ET.SubElement(bbox, "x1").text = str(annotation["bbox"][0])
            ET.SubElement(bbox, "y1").text = str(annotation["bbox"][1])
            ET.SubElement(bbox, "x2").text = str(annotation["bbox"][2])
            ET.SubElement(bbox, "y2").text = str(annotation["bbox"][3])
            ET.SubElement(root, "go_or_stop").text = annotation["go_or_stop"]
            # Add curve points as needed

            tree = ET.ElementTree(root)
            tree.write(f"annotations/{annotation['image']}.xml")

if __name__ == "__main__":
    image_folder = './val100/images'
    anno_folder = './val100/xml'
    annotator = ImageAnnotator(image_folder, anno_folder)
    annotator.root.mainloop()
