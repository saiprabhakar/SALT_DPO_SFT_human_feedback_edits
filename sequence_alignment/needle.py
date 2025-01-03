from .core import OptimalAlignment


class NeedlemanWunsch(OptimalAlignment):
    """
    Needleman Wunsch Alignment object. Takes two sequence objects (seq1 and seq2) and aligns them with the method align.
    """

    def __init__(self, seq1, seq2):
        super(NeedlemanWunsch, self).__init__(seq1, seq2)

    def _add_gap_penalties(self):
        """
        Fills number matrix first row and first column with the gap penalties.
        """
        for i in range(1, len(self.seq1) + 1):
            self._nmatrix[0][i] = self._nmatrix[0][i - 1] + self.smatrix.gap

        for j in range(1, len(self.seq2) + 1):
            self._nmatrix[j][0] = self._nmatrix[j - 1][0] + self.smatrix.gap

    def _get_last_cell_position(self):
        """
        Returns the cell row and column of the last cell in the matrix in which
        the alignment ends. For Needleman-Wunsch this will be the last cell of the matrix,
        for Smith-Waterman will be the cell with the highest score.
        """
        imax = len(self._nmatrix) - 1
        jmax = len(self._nmatrix[0]) - 1
        return imax, jmax

    def _check_best_score(self, diagscore, topscore, leftscore, irow, jcol):
        best_pointer = str()
        best_score = int()
        if diagscore >= topscore:
            if diagscore >= leftscore:
                best_pointer, best_score = ("diag", diagscore)
            else:
                best_pointer, best_score = ("left", leftscore)
        else:
            if topscore > leftscore:
                best_pointer, best_score = ("up", topscore)
            else:
                best_pointer, best_score = ("left", leftscore)

        self._pmatrix[irow + 1][jcol + 1] = best_pointer
        self._nmatrix[irow + 1][jcol + 1] = best_score
        return

    
def get_position_status(al_generated, 
                        al_edited,
                        S_generated_C_weight = 0,
                        S_generated_D_weight = 1,
                        S_generated_S_weight = 1,
                        S_edited_C_weight = 1,
                        S_edited_I_weight = 2,
                        S_edited_S_weight = 2):
    assert len(al_generated)==len(al_edited)
    
    #position_status = []
    S_generated_position_weight = []
    S_edited_position_weight = []
    
    for i in range(len(al_generated)):
        if al_generated[i] == al_edited[i]:
            #position_status.append('C')
            S_generated_position_weight.append(S_generated_C_weight)
            S_edited_position_weight.append(S_edited_C_weight)
        elif al_generated[i] == -100:
            #position_status.append('I')
            S_edited_position_weight.append(S_edited_I_weight)
        elif al_edited[i] == -100:
            #position_status.append('D')
            S_generated_position_weight.append(S_generated_D_weight)
        else:
            #position_status.append('S')
            S_generated_position_weight.append(S_generated_S_weight)
            S_edited_position_weight.append(S_edited_S_weight)
            
    return S_generated_position_weight, S_edited_position_weight