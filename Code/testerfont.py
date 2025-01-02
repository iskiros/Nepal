import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager

font_path = '/Library/Fonts/CMUTypewriter-Bold1.ttc'  # Replace with the actual path
font_prop = font_manager.FontProperties(fname=font_path, size=12)
plt.rcParams['font.family'] = font_prop.get_name()

# Example plot
plt.figure(figsize=(8, 6))
plt.plot([1, 2, 3], [4, 5, 6], label="Example Line")
plt.title("Rain Data in CMU Typewriter Text", fontsize=16)
plt.xlabel("X-axis Label", fontsize=14)
plt.ylabel("Y-axis Label", fontsize=14)
plt.legend()
plt.grid(True)
plt.show()