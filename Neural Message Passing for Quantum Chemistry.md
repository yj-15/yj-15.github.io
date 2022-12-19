# 2. Message Passing Neural Networks

### MPNN이란?
- GNN의 가장 기본이 되는 프레임워크로 node 이웃들의 정보를 이용해 해당 node의 상태를 업데이트하는 형태를 가지는 모든 neural network를 의미함
- 해당 글은 node feature x<sub>v</sub>, edge feature e<sub>vw</sub>를 가지는 undirected graph G를 중점적으로 다룸
- forward pass는 message passing phase와 readout phase, 즉 2개의 phase로 이루어짐
  1) message passing phase  
     
     - 정보를 aggregate하는 message function M<sub>t</sub>, hidden state를 update하는 update function U<sub>t</sub>로 이루어짐  
     - M<sub>t</sub>: node v에 대한 정보를 얻기 위해 정보들을 aggregate하는 message function  
       node v 현재 상태, node v 이웃들(N(v))의 현재 상태, node v와 node의 이웃들을 연결하는 edge들의 정보를 aggregate하여 알아보고 싶은 node의 다음 message를 표현함
     - U<sub>t</sub>: 각 node의 다음 hidden state h<sub>v</sub><sup>t+1</sup>를 얻어진 message m<sub>v</sub><sup>t+1</sup>에 근거해 update하는 function
     
          ![message_passing_phase](https://user-images.githubusercontent.com/120429536/208355404-b9300d5a-d6b3-4b11-84d9-5dc0d9e16c6c.PNG)
     
  2) readout phase
  
      - readout function R: 얻은 node의 hidden state를 활용해 원하는 node의 label 등을 예측, 이는 node state의 permutation에 invariant해야함
      
         ![readout_phase](https://user-images.githubusercontent.com/120429536/208357582-bda0170a-cc26-4ae4-8604-1ba437e37501.PNG)
         

### Convolutional Networks for Learning Molecular Fingerprints (2015)
  - Message Function M(h<sub>v</sub>, h<sub>w</sub>, e<sub>vw</sub>): (h<sub>w</sub>, e<sub>vw</sub>) 
    (,:concatenation)
  - message vector m<sub>v</sub><sup>t+1</sup>: Σh<sub>w</sub><sup>t</sup>,Σe<sub>vw</sub>  
    연결된 node와 edge를 각각 분리하여 합해 edge state와 node state 간의 correlation을 확인하기 어렵다는 단점이 존재함
  - Vertex Update Function U(h<sub>v</sub><sup>t</sup>, m<sub>v</sub><sup>t+1</sup>): σ(H<sub>t</sub><sup>deg(v)</sup>m<sub>v</sub><sup>t+1</sup>)  
    (σ: sigmoid function, deg(v): degree of vertex v, H<sub>t</sub><sup>deg(v)</sup>: vertex degree N일 때, step t에서 학습된 update matrix)
  - Readout Function R: f(Σ softmax(W<sub>t</sub>h<sub>v</sub><sup>t</sup>))  
    (f: neural network, W<sub>t</sub>: 학습된 readout matrix)
    
### Gated Graph Neural Networks (2016)
  - Message Function M(h<sub>v</sub><sup>t</sup>, h<sub>w</sub><sup>t</sup>, e<sub>vw</sub>): A<sub>e<sub>vw</sub></sub>h<sub>w</sub><sup>t</sup>
    (A<sub>e<sub>vw</sub></sub>: 학습된 matrix)
  - Update Function U<sub>t</sub>: GRU(h<sub>v</sub><sup>t</sup>, m<sub>v</sub><sup>t+1</sup>)
    (GRU: Gated Recurrent Unit, weight tying(가중치 공유)을 통해 동일한 update function 사용)
  - Readout Function R: Σ σ(i(h<sub>v</sub><sup>(T)</sup>, h<sub>v</sub><sup>0</sup>))⊙(j(h<sub>v</sub><sup>(T)</sup>))
    (i,j: neural network, ⊙: element wise multiplication)

### Interaction Networks
  - Update Function U: 매 time step마다 적용되는 node level effect를 concatenate한 (h<sub>v</sub>, x<sub>v</sub>, m<sub>v</sub>)을 input으로 하는 neural network
    (x<sub>v</sub>: node v의 외부 영향을 의미하는 external vector를 의미함)
  - Message Function M: (h<sub>v</sub>, h<sub>w</sub>, e<sub>vw</sub>)를 concatenate한 neural network
  - Readout Function R: f(Σ h<sub>v</sub><sup>T</sup>)
    (f: 최종 hidden state인 h<sub>v</sub><sup>T</sup>의 sum을 input으로 하는 neural network)
    
