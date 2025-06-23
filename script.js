

function Fluid(){
    this.gridDim = [100,100];
    this.cellSize = 25;

    this.x = this.gridDim[0]/this.cellSize;
    this.y = this.gridDim[1]/this.cellSize;
    
    this.overrelaxationCoef = 1.5;

    this.points = []
    

    this.setup = function(){
        for (let i = 0; i < (this.x*2)+1; i++){this.points.push([]);}

        for (let i = 0; i < this.x+1; i++){
            for (let j = 0; j < this.y; j++){
                if (i==0 || i==this.x){
                    this.points[i*2][j] = new Point([this.cellSize*i, this.cellSize*(0.5+j)], 0, 0, true); //left point
                }else {
                    this.points[i*2][j] = new Point([this.cellSize*i, this.cellSize*(0.5+j)], 0, 0, false); //left point
                }
            }
        }
        for (let i = 0; i < this.x; i++){
            for (let j = 0; j < this.y+1; j++){
                if (j==0 || j==this.y){
                    this.points[(i*2)+1][j] = new Point([this.cellSize*(0.5+i), this.cellSize*(j)], 0, 0, true); //left point
                }else{
                    this.points[(i*2)+1][j] = new Point([this.cellSize*(0.5+i), this.cellSize*(j)], 0, 0, false); //left point
                }
            }
        }

    }

    this.makeIncompressible = function(iterations){
        for (let iteration = 0; iteration < iterations; iteration++){
            for (let i = 0; i < this.x; i++){
                for (let j = 0; j < this.y; j++){
                    let cellPoints = this.cellToPoints(i,j);

                    let acc = 0
                    let advectingPoints = [];
                    for (let k = 0; k < 4; k++){
                        let PC = cellPoints[k];//point coords
                        let point = this.points[PC[0]][PC[1]];

                        acc += point.vel * (((k==1) || (k==2))? -1 : 1);

                        if (!point.skipAdvection){advectingPoints.push(k);}
                    }
                    for (let l = 0; l < advectingPoints.length; l++){
                        let k = advectingPoints[l];
                        let PC = cellPoints[k];//point coords
                        let point = this.points[PC[0]][PC[1]];

                        point.vel += (acc/advectingPoints.length)* this.overrelaxationCoef * (((k==1) || (k==2))? 1 : -1);
                    }
                }
            }
        }
    }

    this.cellToPoints = function(x,y){
        return [[2*x, y],[(2*x)+1, y],[2*(x+1), y],[(2*x)+1, y+1]]//left, up, right, down
    }
}

function Point(pos, vel, smokeDen, skipAdvection){
    this.pos = pos;
    this.vel = vel;
    this.smokeDen = smokeDen;

    this.skipAdvection = skipAdvection;
}

let fluid = new Fluid();
fluid.setup();
fluid.points[0][1].vel = 10;
fluid.makeIncompressible(20);