# Process Flowchart

```plaintext
 Start  
   ↓  
-------------------------------------------  
|         User Interface (UI) Input        |  
|  - Collects user details & request info  |  
|  - Stores initial data in MongoDB        |  
-------------------------------------------  
   ↓  
[MongoDB] --(Fetch ID)--> [Flask API] --(Store/Query)--> [SQL Database]  
   ↓  
------------------------------------------------  
|            Inspection Agent Assignment       |  
------------------------------------------------  
   ↓  
[Assign Inspection Agent]  
   ├── Match based on **Car Model, Type, Problem**  
   ├── Fetch available agents from DB  
   ├── Assign best-fit agent  
   ↓  
[Slot Booking]  
   ├── Check available slots from DB  
   ├── Consider date, time, service center availability  
   ├── Assign earliest idle slot  
   ↓  
[Generate Inspect Report] (< 1B)  
   ├── Export Formats: `.int`, `.pdf`, `JSON`  
   ↓  
[Wait 1-2 Days]  
   ↓  
------------------------------------------------  
|    Email/SMS Sent: "Inspection Completed"    |  
------------------------------------------------  
   ↓  
------------------------------------------------  
|       Technical & Inventory Agent Assignment |  
------------------------------------------------  
   ↓  
[Assign Technical Agent]  
   ├── Match based on **Car Model, Type, Problem**  
   ├── Fetch available agents from DB  
   ├── Assign best-fit agent  
   ↓  
[Processing Time]  
   ├── If available → **1-2 days**  
   ├── If unavailable → **5 days**  
   ↓  
------------------------------------------------  
|    Email/SMS Sent: "Processing Completed"    |  
------------------------------------------------  
   ↓  
------------------------------------------------  
|    Bill Generation, Payment & Communication  |  
------------------------------------------------  
   ↓  
 End  





# Process Flowchart

```plaintext
 Start  
   ↓  
-------------------------------------------  
|         User Interface (UI) Input        |  
|  - Collects user details & request info  |  
|  - Stores initial data in MongoDB        |  
-------------------------------------------  
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
|    Email/SMS Sent: "Inspection Completed"    |  
------------------------------------------------  
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
|    Email/SMS Sent: "Processing Completed"    |  
------------------------------------------------  
   ↓  
------------------------------------------------  
|    Bill Generation, Payment & Communication  |  
------------------------------------------------  
   ↓  
 End  





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
