import argparse
import ast
import inspect
import torch
import yaml


def collect_mods():
    m={}
    for n,o in inspect.getmembers(torch.nn, inspect.isclass):
        if issubclass(o, torch.nn.Module):
            m[n]=o
    return m


def collect_funcs():
    f={}
    for n,o in inspect.getmembers(torch.nn.functional, inspect.isfunction):
        if not n.startswith("_"):
            f[n]=o
    return f


def count_params(o):
    return sum(p.numel() for p in o.parameters()) if isinstance(o, torch.nn.Module) else 0


def parse_shape(s):
    return tuple(int(i) for i in s.split(','))


class builder:
    def __init__(self, shape, budget, dataset):
        self.start_shape=shape
        self.shape=shape
        self.budget=budget
        self.dataset=dataset
        self.arch=[]
        self.mods=collect_mods()
        self.funcs=collect_funcs()
        self.used=0

    def loop(self):
        while True:
            print("1 Add Layer")
            print("2 Insert Layer")
            print("3 Edit Layer")
            print("4 Delete Layer")
            print("5 View Summary")
            print("0 Finish & Generate YAML")
            k=input()
            if k=="1":
                self.add()
            elif k=="2":
                self.insert()
            elif k=="3":
                self.edit()
            elif k=="4":
                self.delete()
            elif k=="5":
                self.summary()
            elif k=="0":
                self.finish()
                break

    def prompt(self, t, cur=None):
        sig=inspect.signature(t)
        vals={}
        for n,p in sig.parameters.items():
            if n in ("self","input","args","kwargs"):
                continue
            if n in ("in_channels","in_features"):
                vals[n]=self.shape[0] if n=="in_channels" else int(torch.tensor(self.shape).prod().item())
                print(f"{n} set to {vals[n]}")
                continue
            d=None if p.default is inspect._empty else p.default
            if cur and n in cur:
                d=cur[n]
            if t.__name__=="Conv2d" and n=="kernel_size":
                v=input(f"{n} ({d}): ")
                vals[n]=ast.literal_eval(v) if v else d
                continue
            if t.__name__=="Conv2d" and n=="out_channels":
                rem=self.budget-self.used if self.budget else None
                if rem and "kernel_size" in vals:
                    ks=vals["kernel_size"]
                    ks=ks if isinstance(ks,int) else ks[0]*ks[1]
                    hint=rem//(vals["in_channels"]*ks+1)
                    print(f"max {hint}")
                v=input(f"{n} ({d}): ")
                vals[n]=ast.literal_eval(v) if v else d
                continue
            if t.__name__=="Linear" and n=="out_features":
                rem=self.budget-self.used if self.budget else None
                if rem:
                    hint=rem//(vals["in_features"]+1)
                    print(f"max {hint}")
                v=input(f"{n} ({d}): ")
                vals[n]=ast.literal_eval(v) if v else d
                continue
            v=input(f"{n} ({d}): ")
            if v:
                vals[n]=ast.literal_eval(v)
            elif d is not None:
                vals[n]=d
        return vals

    def add(self, idx=None):
        names=list(self.mods.keys())+["functional."+n for n in self.funcs.keys()]
        for i,n in enumerate(names):
            print(i,n)
        c=input()
        if not c.isdigit() or int(c) not in range(len(names)):
            return
        name=names[int(c)]
        if name.startswith("functional."):
            f=self.funcs[name.split(".",1)[1]]
            p=self.prompt(f)
            item={"layer_type":name,"params":p}
        else:
            m=self.mods[name]
            p=self.prompt(m)
            item={"layer_type":name,"params":p}
        tmp=self.arch.copy()
        if idx is None:
            tmp.append(item)
        else:
            tmp.insert(idx,item)
        if self.validate(tmp):
            self.arch=tmp
            self.recalc()

    def insert(self):
        idx=input(f"index 0-{len(self.arch)}:")
        if idx.isdigit():
            self.add(int(idx))

    def edit(self):
        for i,a in enumerate(self.arch):
            print(i,a["layer_type"])
        idx=input()
        if not idx.isdigit() or int(idx) not in range(len(self.arch)):
            return
        i=int(idx)
        a=self.arch[i]
        if a["layer_type"].startswith("functional."):
            f=self.funcs[a["layer_type"].split(".",1)[1]]
            p=self.prompt(f,a["params"])
            item={"layer_type":a["layer_type"],"params":p}
        else:
            m=self.mods[a["layer_type"]]
            p=self.prompt(m,a["params"])
            item={"layer_type":a["layer_type"],"params":p}
        tmp=self.arch.copy()
        tmp[i]=item
        if self.validate(tmp):
            self.arch=tmp
            self.recalc()

    def delete(self):
        for i,a in enumerate(self.arch):
            print(i,a["layer_type"])
        idx=input()
        if not idx.isdigit() or int(idx) not in range(len(self.arch)):
            return
        i=int(idx)
        tmp=self.arch.copy()
        tmp.pop(i)
        if self.validate(tmp):
            self.arch=tmp
            self.recalc()

    def summary(self):
        print("shape",self.shape)
        print("params",self.used)
        for i,a in enumerate(self.arch):
            print(i,a["layer_type"],a["params"])

    def objs_from_arch(self, arch):
        objs=[]
        for a in arch:
            if a["layer_type"].startswith("functional."):
                fn=self.funcs[a["layer_type"].split(".",1)[1]]
                objs.append(lambda x,fn=fn,p=a["params"]: fn(x,**p))
            else:
                cls=self.mods[a["layer_type"]]
                objs.append(cls(**a["params"]))
        return objs

    def validate(self, arch):
        try:
            objs=self.objs_from_arch(arch)
            x=torch.zeros((1,*self.start_shape))
            total=0
            for o in objs:
                x=o(x)
                total+=count_params(o)
            if self.budget and total>self.budget:
                print("budget exceeded")
                return False
        except Exception as e:
            print("invalid",e)
            return False
        return True

    def recalc(self):
        objs=self.objs_from_arch(self.arch)
        x=torch.zeros((1,*self.start_shape))
        total=0
        for o in objs:
            x=o(x)
            total+=count_params(o)
        self.shape=tuple(x.shape[1:])
        self.used=total

    def finish(self):
        name=input("class name:")
        data={
            "model_info":{"class_name":name,"total_parameters":self.used},
            "data_preprocessing":{"dataset_name":self.dataset,"input_shape":list(self.start_shape)},
            "architecture":self.arch
        }
        with open("model_config.yaml","w") as f:
            yaml.dump(data,f)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input-shape",required=True)
    p.add_argument("--budget",type=int)
    p.add_argument("--dataset-name")
    a=p.parse_args()
    b=builder(parse_shape(a.input_shape),a.budget,a.dataset_name)
    b.loop()


if __name__=="__main__":
    main()
