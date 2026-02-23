# Smaill

start date: 10.02

Chaos made character total: 42373
characters:
 !"#%&'()+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_abcdefghijklmnopqrstuvwxyz~
vocab size: 83

<img width="225" height="225" alt="image" src="https://github.com/user-attachments/assets/8312151a-3a35-4b01-b6df-b22188d2a27e" />

v1 (repo): <br/>
no memory 
64 vector size
32 token length
no memory
2000 training loop
output example: wel. nng. Ul Jis.","Theresolivip paloop promeve bestimeatofrace,Sht,"Evetelili,"Ed mbode daninukitmy


v2:
brain cells added - a little logic  + memo to get meaningful output
batch_size = 64


v3: 
no progress ,
ui added ,
opens via localhost 


<img width="954" height="490" alt="image" src="https://github.com/user-attachments/assets/f767a972-ab03-495c-a9a6-26974394dd07" />

to run code: streamlit run app.py

v4: 
trying to solve nonsenseful randomness ,
i tried temp already so top k sampling or top p sampling might solve issue , 
vector size: 64 , 
block size:32 , 
batch size:32 , 
temp:0.8 

v5: 
vector size: 128 ,
block size:128 ,
batch size:64 ,
temp:0.7 ,

v6:
head attention added , 
dataset changed (simple english sentences- he walks home etc) , 
feed forward added: self.blocks = nn.Sequential(MultiHeadAttention(...), FeedForward(...)) , 
there was a mistake while uploading weight in app.py (fixed) , 
load_state_ditch → load_state_dict , 
 
<img width="918" height="362" alt="image" src="https://github.com/user-attachments/assets/26e04464-d577-434f-957e-7c97b58fd5f3" />

why foods fly. 300. He brushes his teethere. 278. The sun feels hair. 86. We eat dish soft song. 298. T
