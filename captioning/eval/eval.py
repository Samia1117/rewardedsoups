__author__ = 'tylin'

import os
from eval.bleu.bleu import Bleu
from eval.meteor.meteor import Meteor
from eval.meteor.reinforce_meteor import Meteor as ReinforceMeteor
from eval.rouge.rouge import Rouge
from eval.cider.cider import Cider
from eval.spice.spice import Spice
from eval.tokenizer.ptbtokenizer import PTBTokenizer
from eval.tokenizer.pythontokenizer import PythonTokenizer
"""
I do not own the rights of this code, I just  modified it according to my needs.
The original version can be found in:
https://github.com/cocodataset/cocoapi
"""

USE_JAVA = True


class COCOEvalCap:
    def __init__(self, dataset_gts_anns, pred_anns, pred_img_ids, get_stanford_models_path=None):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.dataset_gts_anns = dataset_gts_anns
        self.pred_anns = pred_anns
        self.pred_img_ids = pred_img_ids

    def evaluate(self, bleu=True, rouge=True, cider=True, spice=True, meteor=True, verbose=True):
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for imgId in self.pred_img_ids:
            gts[imgId] = self.dataset_gts_anns[imgId]
            res[imgId] = self.pred_anns[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        print("load tokenization")
        if not USE_JAVA:
            tokenizer = PythonTokenizer()
        else:
            tokenizer = PTBTokenizer()
        print("begin tokenization")
        gts = tokenizer.tokenize(gts)
        print("end tokenization gts")
        res = tokenizer.tokenize(res)
        print("end tokenization res")
        # =================================================
        # Set up scorers
        # =================================================
        #if verbose:
        #    print('setting up scorers...')
        scorers = []
        if cider:
            scorers.append((Cider(), "CIDEr"))
        if bleu:
            scorers.append((Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]))
        if rouge:
            scorers.append((Rouge(), "ROUGE_L"))
        if spice and USE_JAVA:
            scorers.append((Spice(), "SPICE"))
        if meteor:
            if USE_JAVA:
                scorers.append((Meteor(), "METEOR"))
            else:
                scorers.append((ReinforceMeteor(), "METEOR"))

        # =================================================
        # Compute scores
        # =================================================
        return_scores = []
        for scorer, method in scorers:
            print(scorer)
            if verbose:
                # print('computing %s score...'%(scorer.method()))
                pass
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    if verbose:
                        # print("%s: %0.3f"%(m, sc))
                        pass
                    return_scores.append((m, round(sc, 4)))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                if verbose:
                    # print("%s: %0.3f"%(method, score))
                    pass
                return_scores.append((method, round(score, 4)))
        self.setEvalImgs()

        return return_scores

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
