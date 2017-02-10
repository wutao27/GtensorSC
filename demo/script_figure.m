% ---------------------- script for plotting openFlight data ---------------------------
%% load the openFlight rectangular tensor, and clustering results
tempRT= load('../data/openFlight/rectangle.txt');
T(:,1) = tempRT(:,2) - 539;
T(:,2) = tempRT(:,3) - 539;
T(:,3) = tempRT(:,1);
T(:,4) = tempRT(:,4);
clear tempRT

order_airline = load('../data/openFlight/result_airline.txt');
order_airport = load('../data/openFlight/result_airport.txt');
% The above result files are the algorithm results. Each line is an airline/airport id sorted based on the algorithm.
[~,p_airline] = sort(order_airline);
[~,p_airport] = sort(order_airport);

%% plot the 3D openflight data when randomly sorted
aa = rand(3478-539,1);
bb = rand(539,1);
[~,pa] = sort(aa);
[~,pb] = sort(bb);

n1 = max(T(:,1));n2 = max(T(:,2));n3 = max(T(:,3));
shrink1 = 10; shrink2 = 10;shrink3 = 5;
% skrink is used to combine several neighbor nodes together to decrease the tensor size, as the tensor is too big to plot
m1 = ceil(n1/shrink1); m2 = ceil(n2/shrink2);m3 = ceil(n3/shrink3);
x = zeros(m1,m2,m3);
for i=1:size(T,1)
    x(ceil(pa(T(i,1))/shrink1),ceil(pa(T(i,2))/shrink2),ceil(pb(T(i,3))/shrink3)) = x(ceil(pa(T(i,1))/shrink1),ceil(pa(T(i,2))/shrink2),ceil(pb(T(i,3))/shrink3)) + T(i,4);
end

h = vol3d('cdata',x,'texture','3D');

view(3)
alphamap(5 .* alphamap);

xt = {'500', '1500', '2500'};
set(gca,'XTick',[ 50,150,250]);
set(gca,'XTickLabel',xt)

set(gca,'YTick',[ 50,150,250]);
set(gca,'YTickLabel',xt)

zt = {'100','300','500'};
set(gca,'ZTick',[ 20,60,100]);
set(gca,'ZTickLabel',zt)

xlabel('Airports');
ylabel('Airports');
zlabel('Airlines');


%% plot the 3D openflight data when sorted by clustering results
clf
n1 = max(T(:,1));n2 = max(T(:,2));n3 = max(T(:,3));
shrink1 = 10; shrink2 = 10;shrink3 = 5;
m1 = ceil(n1/shrink1); m2 = ceil(n2/shrink2);m3 = ceil(n3/shrink3);
y = zeros(m1,m2,m3);
for i=1:size(T,1)
    y(ceil(p_airport(T(i,1))/shrink1),ceil(p_airport(T(i,2))/shrink2),ceil(p_airline(T(i,3))/shrink3)) = y(ceil(p_airport(T(i,1))/shrink1),ceil(p_airport(T(i,2))/shrink2),ceil(p_airline(T(i,3))/shrink3)) + T(i,4);
end

h = vol3d('cdata',y,'texture','3D');
view(3)
alphamap(5 .* alphamap);

xt = {'500', '1500', '2500'};
set(gca,'XTick',[ 50,150,250]);
set(gca,'XTickLabel',xt)

set(gca,'YTick',[ 50,150,250]);
set(gca,'YTickLabel',xt)

zt = {'100','300','500'};
set(gca,'ZTick',[ 20,60,100]);
set(gca,'ZTickLabel',zt)

xlabel('Airports');
ylabel('Airports');
zlabel('Airlines');


% ---------------------- script for plotting Enron example ---------------------------
%% load enron data
X = load('../data/Enron/tensor_enron.txt');
A = X((X(:,1)==373),:);
xx = zeros(185,1);
for i=1:size(A,1)
    if A(i,2)>184 && A(i,2)<370 
        xx(A(i,2)-184) = xx(A(i,2)-184)+A(i,5);
    end
end

A = X((X(:,1)==379),:);
yy = zeros(185,1);
for i=1:size(A,1)
    if A(i,2)>184 && A(i,2)<370 
        yy(A(i,2)-184) = yy(A(i,2)-184)+A(i,5);
    end
end

A = X((X(:,1)==394),:);
zz = zeros(185,1);
for i=1:size(A,1)
    if A(i,2)>184 && A(i,2)<370 
        zz(A(i,2)-184) = zz(A(i,2)-184)+A(i,5);
    end
end

%% plot the enron example
clf
plot(1:185,xx,'LineWidth',1)
hold on
plot(1:185,yy,'LineWidth',1)
hold on
plot(1:185,zz,'LineWidth',1)

box off
xlim([90,180])

yl = ylim;
yl(2) = yl(2)*0.9;
line(107*[1,1],yl,'Color','k','LineWidth',0.5)
text(107, yl(2), {'CEO Skilling','appointed'},...
    'VerticalAlignment','bottom','HorizontalAlignment','center')

line(107*[1,1],yl*0.05,'Color','r','LineWidth',2.5)


line(107*[1,1],yl*0.5,'Color','k','LineWidth',0.5)
text(107, yl(2)*0.5, {' '},...
    'VerticalAlignment','bottom','HorizontalAlignment','center')

line(107*[1,1],yl*0.05,'Color','r','LineWidth',2.5)

% line(109*[1,1],yl*0.8,'Color','k','LineWidth',0.2)
% text(109, yl(2)*0.8, {' '},...
%     'VerticalAlignment','bottom','HorizontalAlignment','center')

line(109*[1,1],yl*0.05,'Color','r','LineWidth',2.5)

line(113*[1,1],yl*0.65,'Color','k','LineWidth',0.5)
text(113, yl(2)*0.65, {'CEO Skilling','Started'},...
    'VerticalAlignment','bottom','HorizontalAlignment','Left')

line(113*[1,1],yl*0.05,'Color','r','LineWidth',2.5)