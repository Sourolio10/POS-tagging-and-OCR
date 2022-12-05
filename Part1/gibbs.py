import traceback
import numpy as np
class Gibbs:
    
    def __init__(self):
        self.POS = None
        self.em_prob_proc = None
        self.tr_prob = None
        self.tr_prob2 = None
    
    def train(self,data):
        POS = ['adj','adv','adp','conj','det','noun','num','pron','prt','verb','x','.']
        em_prob = {}
        tr_prob = {}
        tr_prob2 = {}
        pos_cnt = {}
        for row in data:
            words = row[0]
            pos = row[1]
            cnt=0
            for w,p in zip(words,pos):
                if p not in em_prob:
                    em_prob[p]={}
                if w not in em_prob[p]:
                    em_prob[p][w]=0
                em_prob[p][w]+=1
                if p not in pos_cnt:
                    pos_cnt[p]=0
                pos_cnt[p]+=1
                if cnt<len(words)-1:
                    next_pos = pos[cnt+1]
                    if p not in tr_prob:
                        tr_prob[p]={}
                    if next_pos not in tr_prob[p]:
                        tr_prob[p][next_pos]=0
                    tr_prob[p][next_pos]+=1
                if cnt<len(words)-2:
                    next_pos = pos[cnt+2]
                    if p not in tr_prob2:
                        tr_prob2[p]={}
                    if next_pos not in tr_prob2[p]:
                        tr_prob2[p][next_pos]=0
                    tr_prob2[p][next_pos]+=1
                cnt+=1

        for i,j in em_prob.items():
            for k,l in j.items():
                em_prob[i][k]=l/pos_cnt[i]
        for i,j in tr_prob.items():
            for k,l in j.items():
                tr_prob[i][k]=l/pos_cnt[i]
        for i,j in tr_prob2.items():
            for k,l in j.items():
                tr_prob2[i][k]=l/pos_cnt[i]

        vocab = list(em_prob.keys())

        em_prob_proc  ={}
        for i,j in em_prob.items():
            for k,l in j.items():
                if k not in em_prob_proc:
                    em_prob_proc[k]={}
                if i not in em_prob_proc[k]:
                    em_prob_proc[k][i]=0
                em_prob_proc[k][i]=l

        for i,j in em_prob_proc.items():
            for p in POS:
                if p not in j:
                    em_prob_proc[i][p]=1e-8

        for i in POS:
            for j in POS:
                if i not in tr_prob:
                    tr_prob[i]={}
                if j not in tr_prob[i]:
                    tr_prob[i][j]=1e-8

        for i in POS:
            for j in POS:
                if i not in tr_prob2:
                    tr_prob2[i]={}
                if j not in tr_prob2[i]:
                    tr_prob2[i][j]=1e-8 
        
        self.POS = POS
        self.em_prob_proc = em_prob_proc
        self.tr_prob = tr_prob
        self.tr_prob2 = tr_prob2

    def run_gibbs(self,words):
        try:
            rng = np.random.default_rng()
            mtx = rng.choice(self.POS,len(words)).tolist()
            val_mtx = [1e-8]*len(words)
            mp_vl_log = []
            for itr in range(500):
                for w_idx,w in enumerate(words):
                    min_val = np.inf
                    best_p = 'x'
                    for p in self.POS:

                        if w in self.em_prob_proc:
                            emp = self.em_prob_proc[w][p]
                            if w_idx>0:
                                emp_prev = self.em_prob_proc[w][mtx[w_idx-1]]
                        else:
                            emp = 1e-8
                            emp_prev=1e-8

                        if w_idx>=2:
                            val = -np.log(self.tr_prob[mtx[w_idx-1]][p])-np.log(val_mtx[w_idx-1])-np.log(self.tr_prob2[mtx[w_idx-2]][p])-np.log(val_mtx[w_idx-2])-np.log(emp)-np.log(emp_prev)
                        elif w_idx==1:
                            val = -np.log(self.tr_prob[mtx[w_idx-1]][p])-np.log(val_mtx[w_idx-1])-np.log(emp)-np.log(emp_prev)
                        elif w_idx==0:
                            val = -np.log(emp)
                        if val < min_val:
                            min_val=val
                            best_p=p
                    mtx[w_idx]=best_p
                    val_mtx[w_idx]=min_val

                    mp_val = np.sum(val_mtx)
                    mp_vl_log.append(mp_val)

                # if itr%50==0:
                #     if np.std(mp_vl_log[-20:])<1e-5:
                #         break
        except Exception as e:
            traceback.print_exc()
            print(words)
            print(itr)
            print(w)
            print(p)
            print(mtx)
            print(val_mtx)
            assert False
        return mtx,val_mtx,mp_vl_log
