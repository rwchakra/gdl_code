from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
class LabelGenerate():
  def __init__(self, x, h, edges, edge_attr, batch_size, n_nodes, graphs):
    # Returns (x', h'), where x' is invariant to E(n), and h' is equivariant to E(n)
    #x' = x + g(x,h,G) = x + p(h)* A * phi(x) * f_1(x)
    #h = p(h) * A * f_2(x)


    self.x = x
    self.h = h
    self.edges = edges
    self.edge_attr = edge_attr
    self.batch_size = batch_size
    self.n = n_nodes
    self.graphs = graphs
    #self.W = 

  def x_1(self, feat):
    dmatrix = np.empty(shape=(self.batch_size , self.n, self.x.shape[1]))
    c=0
    for i in range(0, self.x.shape[0], self.n):

      g = self.graphs[c]
      graph = self.x[i:i+self.n]
      s = graph[:, :, None] - graph[:, :, None].T
      s_swap = np.array(np.swapaxes(s, 1, 2))
      #f_x = (np.array(feat[i:i+self.n].sum(1)).reshape((self.n, 1, 1)) * s_swap * g.reshape((self.n,self.n,1))).sum(1)
      f_x = (np.array(feat[i:i+self.n].sum(1)).reshape((self.n, 1, 1)) * s_swap ).sum(1)
      dmatrix[c] = f_x
      c+=1

    return self.x + dmatrix.reshape(-1, dmatrix.shape[2])

    

  def h_1(self, feat, type_f2 = 'distance'):

    if type_f2 == 'volume':
    
      volume = np.empty(shape = (self.batch_size, self.n, 1))
      ones = np.ones(shape = (self.n, 1))
      c = 0
      for i in range(0, self.x.shape[0], self.n):
        g = self.graphs[c]
        coords = self.x[i:i+self.n]
        #print(coords)
        vol = ConvexHull(coords).volume
        vol_array = ones * np.array(vol)
        f_2x = np.array((feat[i:i+self.n].sum(1) * g @ vol_array))
        volume[c] = f_2x
        c+=1
      h_1 = volume

    elif type_f2 == 'area':

      area = np.empty(shape = (self.batch_size, self.n, 1))
      ones = np.ones(shape = (self.n, 1))
      c = 0
      for i in range(0, self.x.shape[0], self.n):
        g = self.graphs[c]
        coords = self.x[i:i+self.n]
        #print(coords)
        area_coord = ConvexHull(coords).area
        area_array = ones * np.array(area_coord)
        f_2x = np.array((feat[i:i+self.n].sum(1) * g @ area_array))
        area[c] = f_2x
        c+=1
      h_1 = area

    elif type_f2 == 'distance':
      
      dist = np.empty(shape = (self.batch_size, self.n, 1))
      c = 0
      
      for i in range(0, self.x.shape[0], self.n):
        g = self.graphs[c]
        coords = self.x[i:i+self.n]
        d = squareform(pdist(coords))
        f_2x = (g * d * np.array(feat[i:i+self.n].sum(1))).sum(1)
        dist[c] = f_2x.reshape(-1, 1)
        c+=1
        
      h_1 = dist

    elif type_f2 == 'angles':
      angles = np.empty(shape = (self.batch_size, self.n, 1))
      c = 0
      for i in range(0, self.x.shape[0], self.n):
        g = self.graphs[c]
        coords = self.x[i:i+self.n]
        d = np.zeros((self.n, self.n, self.n))
        for i in range(self.n):
          for j in range(self.n):
            for k in range(self.n):
              u = coords[i,:]
              v = coords[j,:]
              w = coords[k,:]
              a = u - v
              b = w - v
              angle_ijk = np.arccos(a.dot(b)/(np.linalg.norm(a)*np.linalg.norm(b)))
              d[i,j,k] = angle_ijk
        d = d.sum(2)
        f_2x = (g * d * np.array(feat[i:i+self.n].sum(1))).sum(1)
        angles[c] = f_2x.reshape(-1, 1)
        c+=1
      h_1 = angles

    return h_1.reshape(-1, h_1.shape[2])



  def p(self):
    #simply exponential
    #return np.exp(self.h)
    return self.h
    #return np.ones(shape=self.h.shape)

  def get_labels(self, type_f2 = 'distance'):

    p = self.p()
    x_1 = self.x_1(p)
    h_1 = self.h_1(p, type_f2)
    #Standard Normalization
    #print(type(x_1))
    #x_1 = (x_1 - torch.mean(x_1))/torch.std(x_1)
    #h_1 = (h_1 - np.mean(h_1))/np.std(h_1)
    #Frobenius Normalization
    return x_1, h_1