# Smaill </br>
small language model written and train fully locally </br>

start date: 10.02

Chaos made character total: 42373
characters: <br/>
 !"#%&'()+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_abcdefghijklmnopqrstuvwxyz~ <br/>
vocab size: 83 <br/>

  <img width="310" height="170" alt="image" src="https://github.com/user-attachments/assets/eefa04bf-c945-45b1-836d-3bef392e9aa4" /><br/>

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

27.02 codes took to much time to train due to hardware issues
trying to run it on gpu, until that 23.02 is last one

v7: 01.03 runs on gpu, better output quality , ++batch ++head , will try again w bigger dataset </br>

Vocabulary size: 83
Total tokens: 57224
Model parameters: 0.35M
Step 0: loss 4.7196 | Sample:  0pCel te2Nw&XnJrDOMa4z[qk3e#(g;cCg'Bnm!ltaY-u:HEY~ 
Step 200: loss 2.5153 | Sample:  Grwavin Ithts ly. She bof 2287. ther t out inghero
(...)
Step 29600: loss 1.1040 | Sample:  I buys of the plays witter. 426. She closes the st
Step 29800: loss 1.0925 | Sample:  of Zepperonic (1994)" "Yo canclin shiki bird. 3142
Model trained & weights saved...

</br>
13.03: larger dataset, 30000 epochs  (0.36 parameters)</br>
Teleman. 80. The stering tall have young from and distand down of the lonce, and charp
</br>

