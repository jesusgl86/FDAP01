function M=diffmat(v)
[X,Y]=meshgrid(v,v);
M=X-Y;
M=flipud(M);
end