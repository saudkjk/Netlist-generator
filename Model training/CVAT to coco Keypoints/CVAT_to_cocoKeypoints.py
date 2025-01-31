import os
from xml.dom import minidom

# Directory to save the output files
out_dir = './out'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Parse the XML file
file = minidom.parse('annotations.xml')

# Fixed label mapping for resistor and transistor

label_mapping = { "Resistor": 0, "Capacitor": 1, "Inductor": 2, "Transistor_BJT": 3, "Transistor_MOSFET": 4, "Voltage_src": 5, "Current_src": 6, "GND": 7}
label_counter = 7

# Processing each image
images = file.getElementsByTagName('image')

for image in images:
    width = int(image.getAttribute('width'))
    height = int(image.getAttribute('height'))
    name = image.getAttribute('name')
    points_elements = image.getElementsByTagName('points')
    bboxes = image.getElementsByTagName('box')  # Retrieve all <box> elements

    # Create a new label file for each image
    label_file_path = os.path.join(out_dir, name[:-4] + '.txt')
    with open(label_file_path, 'w') as label_file:

        # Iterate over all bounding boxes
        for bbox in bboxes:
            bbox_label = bbox.getAttribute("label")

            # Ensure the label is mapped
            if bbox_label not in label_mapping:
                label_mapping[bbox_label] = label_counter
                label_counter += 1

            label_id = label_mapping[bbox_label]

            # Retrieve bounding box coordinates
            xtl = float(bbox.getAttribute('xtl'))
            ytl = float(bbox.getAttribute('ytl'))
            xbr = float(bbox.getAttribute('xbr'))
            ybr = float(bbox.getAttribute('ybr'))
            w = xbr - xtl
            h = ybr - ytl

            # Normalize bounding box (center x, center y, width, height)
            bbox_center_x = (xtl + (w / 2)) / width
            bbox_center_y = (ytl + (h / 2)) / height
            bbox_norm_w = w / width
            bbox_norm_h = h / height

            # Write bbox data
            label_file.write(f"{label_id} {bbox_center_x} {bbox_center_y} {bbox_norm_w} {bbox_norm_h} ")

            # Match points to the current bounding box
            matched_points = []
            for points in points_elements:
                points_label = points.getAttribute("label")

                if points_label == bbox_label:  # Match labels
                    points_data = points.getAttribute('points')
                    points_list = points_data.split(';')

                    for point in points_list:
                        p1, p2 = map(float, point.split(','))

                        # Check if the point is inside the bounding box
                        if xtl <= p1 <= xbr and ytl <= p2 <= ybr:
                            # Normalize point coordinates
                            norm_p1 = p1 / width
                            norm_p2 = p2 / height
                            matched_points.append((norm_p1, norm_p2, 1))  # 1 = visible

            # Add dummy points if there are fewer than 3 keypoints
            while len(matched_points) < 3:
                matched_points.append((0.0, 0.0, 0))  # Dummy point with visibility 0
            
            # Write keypoints data
            for i, (norm_p1, norm_p2, visibility) in enumerate(matched_points[:3]):  # Ensure only 3 keypoints
                label_file.write(f"{norm_p1} {norm_p2} {visibility}")
                if i < len(matched_points[:3]) - 1:
                    label_file.write(" ")
                else:
                    label_file.write("\n")  # New line for the next bbox

print("Output files created successfully in:", out_dir)
