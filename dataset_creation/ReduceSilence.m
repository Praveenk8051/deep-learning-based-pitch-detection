function Notes = ReduceSilence(Notes)
n1 = Notes(2:end,5);
n2 = Notes(1:end-1,6);
i = 1;
j = 1;
rows = [];
while j < length(n1)
    if n1(j)-n2(j) > 0.0001 && n1(j)-n2(j) < 0.5
        d = n1(j)-n2(j);
        n1(j) = n2(j);
%         n2(j+1:end) = n2(j+1:end)-d;
%         n1(j+1:end) = n1(j+1:end)-d;
    elseif n1(j)-n2(j) > 0.5
        d = n1(j)-n2(j);
        n1(j) = n2(j)+0.1;
%         n2(j+1:end) = n2(j+1:end)-d + 0.1;
%         n1(j+1:end) = n1(j+1:end)-d + 0.1;
    end
    j = j+1;
end
Notes(2:end,5) = n1;
Notes(1:end-1,6) = n2;
end