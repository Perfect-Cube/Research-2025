# @title Display Sketchfab Embed

from IPython.display import display, HTML

# The HTML code provided by the user (Sketchfab embed)
# Added width and height attributes to the iframe for better default sizing in Colab
sketchfab_embed_html = """
<div class="sketchfab-embed-wrapper">
    <iframe title="Scania truck" frameborder="0" allowfullscreen mozallowfullscreen="true" webkitallowfullscreen="true"
            allow="autoplay; fullscreen; xr-spatial-tracking" xr-spatial-tracking
            execution-while-out-of-viewport execution-while-not-rendered web-share
            width="800" height="600"  # <-- Added width/height for better default size
            src="https://sketchfab.com/models/59889032d0ad457c81d7e058c79eedf8/embed">
    </iframe>
    <p style="font-size: 13px; font-weight: normal; margin: 5px; color: #4A4A4A;">
        <a href="https://sketchfab.com/3d-models/scania-truck-59889032d0ad457c81d7e058c79eedf8?utm_medium=embed&utm_campaign=share-popup&utm_content=59889032d0ad457c81d7e058c79eedf8"
           target="_blank" rel="nofollow" style="font-weight: bold; color: #1CAAD9;"> Scania truck </a>
        by <a href="https://sketchfab.com/PusztaiAndras?utm_medium=embed&utm_campaign=share-popup&utm_content=59889032d0ad457c81d7e058c79eedf8"
              target="_blank" rel="nofollow" style="font-weight: bold; color: #1CAAD9;"> PAndras </a>
        on <a href="https://sketchfab.com?utm_medium=embed&utm_campaign=share-popup&utm_content=59889032d0ad457c81d7e058c79eedf8"
              target="_blank" rel="nofollow" style="font-weight: bold; color: #1CAAD9;">Sketchfab</a>
    </p>
</div>
"""

print("Displaying Sketchfab embed below:")

# Display the HTML content in the Colab output cell
display(HTML(sketchfab_embed_html))

# You might still want to display the license details explicitly from the license.txt file
# as good practice, although the embed includes attribution links.
license_text = """
--- License Information (from license.txt) ---
Model Information:
* title:	Scania truck
* source:	https://sketchfab.com/3d-models/scania-truck-59889032d0ad457c81d7e058c79eedf8
* author:	PAndras (https://sketchfab.com/PusztaiAndras)

Model License:
* license type:	CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)
* requirements:	Author must be credited. Commercial use is allowed.

If you use this 3D model in your project be sure to copy paste this credit wherever you share it:
This work is based on "Scania truck" (https://sketchfab.com/3d-models/scania-truck-59889032d0ad457c81d7e058c79eedf8) by PAndras (https://sketchfab.com/PusztaiAndras) licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)
"""
print(license_text)
