import torch

class CFG:
    """
    Context-Free Grammar implementation for hierarchical language modeling.
    
    Parameters:
    - ns: [ns_0, ns_1, ..., ns_L] - number of symbols at each level (L+1 integers)
    - nr: [nr_0, nr_1, ..., nr_{L-1}] - number of rules per symbol at each level (L integers)  
    - T: [T_0, T_1, ..., T_{L-1}] - length of rules at each level (L integers)
    - L: number of levels
    
    Level structure:
    - Level 0: class labels (ns_0 labels)
    - Level 1: Tensors of shape (T_0,) with level-1 symbols
    - ...
    - Level L: Tensors of shape (T_0,T_1,...,T_{L-1}) with level-L symbols
    
    Rules:
    - rules[l]: shape (ns_l, nr_l, T_l) with entries in {0,1,...,ns_{l+1}-1}
    """

    def __init__(self, L, ns, nr, T):
        assert L == len(nr) and L == len(T) and L == len(ns) - 1
        self.ns = ns
        self.nr = nr
        self.T = T
        self.L = L
        self.rules = []
        for l in range(L):
            self.rules.append(torch.randint(0, ns[l + 1], size=(ns[l], nr[l], T[l])))

    def expand_symbols_one_level(self, S, l):
        """Expand symbols from level l to level l+1 using rules[l]"""
        RND = torch.randint(0, self.nr[l], size=S.shape)
        return self.rules[l][S, RND]

    def expand_symbols(self, S, l=0):
        """Expand from level l to level L"""
        for lev in range(l, self.L):
            S = self.expand_symbols_one_level(S, lev)
        return S

    def sample(self, nspl):
        """Generate nspl data points per class with labels"""
        labels = torch.arange(self.ns[0]).repeat_interleave(nspl, dim=0)
        S = self.expand_symbols(labels)
        return S, labels

    def sample_flattened(self, nspl):
        """Generate nspl flattened data points per class with labels"""
        labels = torch.arange(self.ns[0]).repeat_interleave(nspl, dim=0)
        S = self.expand_symbols(labels)
        return S.view((self.ns[0], nspl, -1)), labels

    def closest_rule(self, seq, l):
        """Find closest rule at level l to sequence seq (Hamming distance)"""
        seq = seq.view(1, 1, -1)
        rules_l = self.rules[l].view(-1, self.T[l])
        distances = (rules_l != seq).sum(dim=1)
        min_idx = distances.argmin()
        symbol_idx = min_idx // self.nr[l]
        rule_idx = min_idx % self.nr[l]
        return symbol_idx.item(), rule_idx.item(), distances[min_idx].item()

    def collapse_one_level(self, S, l):
        """Collapse from level l+1 to level l"""
        shape = S.shape
        S_flat = S.view(-1, self.T[l])
        collapsed = torch.zeros(S_flat.shape[0], dtype=torch.long)
        errors = torch.zeros(S_flat.shape[0], dtype=torch.long)
        
        for i, seq in enumerate(S_flat):
            symbol_idx, _, error = self.closest_rule(seq, l)
            collapsed[i] = symbol_idx
            errors[i] = error
            
        new_shape = shape[:-1]
        return collapsed.view(new_shape), errors.view(new_shape)

    def collapse_and_get_err(self, S):
        """Collapse from level L to level 0 and return errors at each level"""
        errors = []
        for l in range(self.L - 1, -1, -1):
            S, err = self.collapse_one_level(S, l)
            errors.append(err)
        return S, errors[::-1]

    def frac_of_gramatically_correct_sentences(self, sentences):
        """Compute fraction of grammatically correct sentences"""
        correct = 0
        for sentence in sentences:
            _, errors = self.collapse_and_get_err(sentence)
            if all(err.sum() == 0 for err in errors):
                correct += 1
        return correct / len(sentences)
