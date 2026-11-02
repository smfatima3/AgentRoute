"""One isolated evaluator process. Its parent enforces a wall-clock deadline."""
import json, sys
from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig
def main():
    request=json.load(sys.stdin)
    config=[LatexExtractionConfig(),ExprExtractionConfig()]
    def parsed(s):
        # The parent enforces the full deadline. Avoid nested Windows multiprocessing.
        return parse('$'+s+'$',extraction_config=config,parsing_timeout=None)
    prediction=parsed(request['prediction'])
    golds=[parsed(s) for s in request['answers']]
    ok=bool(prediction)
    gold_ok=any(bool(g) for g in golds)
    print(json.dumps({'correct':bool(ok and any(verify(g,prediction,timeout_seconds=None) for g in golds if g)),
                      'prediction_parsed':ok,'gold_parsed':gold_ok,
                      'symbolic_prediction':any(not isinstance(p,str) for p in prediction),
                      'symbolic_gold':any(not isinstance(g,str) for values in golds for g in values)}))

if __name__=='__main__':main()
