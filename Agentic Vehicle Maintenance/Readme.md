# Process Flowchart

```plaintext
 Start  
   ↓  
[MongoDB] --(Fetch ID)--> [Flask API] --(Store/Query)--> [SQL Database]  
   ↓  
------------------------------------------------  
|               Inspection Agent                |  
------------------------------------------------  
   ↓  
[Generate Inspect Report] (< 1B)  
   ├── Export Formats: `.int`, `.pdf`, `JSON`  
   ↓  
[Wait 1-2 Days]  
   ↓  
------------------------------------------------  
|       Inventory & Technical Agent            |  
------------------------------------------------  
   ↓  
[Processing Time]  
   ├── If available → **1-2 days**  
   ├── If unavailable → **5 days**  
   ↓  
------------------------------------------------  
|    Bill Generation, Payment & Communication  |  
------------------------------------------------  
   ↓  
 End  
