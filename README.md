# Smaill

start date: 10.02

Chaos made character total: 42373
characters:
 !"#%&'()+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_abcdefghijklmnopqrstuvwxyz~
vocab size: 83 <br/>

<img align="center" width="225" height="225" alt="image" src="https://github.com/user-attachments/assets/8312151a-3a35-4b01-b6df-b22188d2a27e" /> <br/>

v1 (repo): <br/>
no memory <br/>
64 vector size<br/>
32 token length<br/>
no memory<br/>
2000 training loop<br/><br/>
output example: wel. nng. Ul Jis.","Theresolivip paloop promeve bestimeatofrace,Sht,"Evetelili,"Ed mbode daninukitmy<br/>


v2:<br/>
brain cells added - a little logic  + memo to get meaningful output<br/>
batch_size = 64<br/>

<br/>
v3: <br/>
no progress ,<br/>
ui added ,<br/>
opens via localhost <br/>

<br/><br/>
<img width="954" height="490" alt="image" src="https://github.com/user-attachments/assets/f767a972-ab03-495c-a9a6-26974394dd07" />
<br/>
to run code: streamlit run app.py
<br/>
v4: <br/>
trying to solve nonsenseful randomness ,<br/>
i tried temp already so top k sampling or top p sampling might solve issue , <br/>
vector size: 64 , <br/>
block size:32 , <br/>
batch size:32 , <br/>
temp:0.8 <br/>
<br/>
v5: <br/>
vector size: 128 ,<br/>
block size:128 ,<br/>
batch size:64 ,<br/>
temp:0.7 ,<br/>
<br/>
v6:<br/>
head attention added , <br/>
dataset changed (simple english sentences- he walks home etc) , <br/>
feed forward added: self.blocks = nn.Sequential(MultiHeadAttention(...), FeedForward(...)) , <br/>
there was a mistake while uploading weight in app.py (fixed) , <br/>
load_state_ditch → load_state_dict , <br/>
 <br/>
<img width="918" height="362" alt="image" src="https://github.com/user-attachments/assets/26e04464-d577-434f-957e-7c97b58fd5f3" />

why foods fly. 300. He brushes his teethere. 278. The sun feels hair. 86. We eat dish soft song. 298. T
