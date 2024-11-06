///// Capacitated Vehicle Routing Problem 
///// with a Heterogeneous Vehicle Fleet and Restrictions on the Number of Vehicles Utilized 
///// (H-CVRP-MV)
///// Dr. Racem Mellouli - M2 GOL (Software Applications for Operations Optimization)

// Note: There exist various vehicle usage restrictions, such as daily mileage limits 
// and workday duration, in addition to vehicle capacity. Specifically, this model focuses on 
// limiting the number of vehicles utilized among the available fleet.


#include <ilcplex/ilocplex.h>
#include <iostream>
ILOSTLBEGIN
#include <vector>
void disp1(vector<int> x) {for (const auto& val : x) cout << val << " ";}
void disp2(vector<vector<double>> x) { cout << endl; for (const auto& row : x) { for (const auto& val : row) cout << "\t" << val; cout << endl; } }

int main() {
    IloEnv env;
    try {
        IloModel model(env);

        // Problem parameters
        const int n = 10;      // Number of nodes (including depot)
        const int m = 4;       // Number of vehicles
        const int MAXV = 4;    // Maximum number of vehicles allowed
        const int MINV = 0;    // Minimum number of vehicles allowed

        vector<int> q = { 0, 3, 4, 2, 7, 5, 3, 2, 7, 2 }; // Demand at each node
        vector<int> Q = { 10, 10, 15, 20 }; // Capacity of each vehicle

        vector<vector<double>> c = {// Distance (cost) matrix
            { 0.0, 8.9, 7.1, 10.4, 26.5, 15.3, 10.5, 39.4, 27.0, 13.8 },
            { 8.9, 0.0, 10.2, 19.2, 19.7, 9.4, 19.1, 35.0, 32.4, 18.4 },
            { 7.1, 10.2, 0.0, 12.3, 30.0, 19.3, 15.7, 33.0, 22.2, 20.9 },
            { 10.4, 19.2, 12.3, 0.0, 36.5, 25.3, 6.7, 44.2, 21.0, 17.0 },
            { 26.5, 19.7, 30.0, 36.5, 0.0, 11.2, 33.5, 47.0, 52.2, 26.0 },
            { 15.3, 9.4, 19.3, 25.3, 11.2, 0.0, 22.7, 42.2, 41.4, 17.0 },
            { 10.5, 19.1, 15.7, 6.7, 33.5, 22.7, 0.0, 48.7, 27.7, 10.8 },
            { 39.4, 35.0, 33.0, 44.2, 47.0, 42.2, 48.7, 0.0, 37.3, 52.7 },
            { 27.0, 32.4, 22.2, 21.0, 52.2, 41.4, 27.7, 37.3, 0.0, 37.9 },
            { 13.8, 18.4, 20.9, 17.0, 26.0, 17.0, 10.8, 52.7, 37.9, 0.0 }
        };
        // Display the problem data
        cout << "Data for Capacitated Vehicle Routing Problem\n\twith Heterogenuous Fleet and Used Vehicle Number Restrictions : (H-CVRP-MV)\n";
        cout << "\n(n) Number of nodes including depot: " << n;
        cout << "\n(m) Number of available vehicles : " << m;
        cout << "\n(q) Demand at each node: "; disp1(q);
        cout << "\n(Q) Vehicle capacity:  "; disp1(Q);
        cout << "\n(c) Distance or cost matrix:"; disp2(c);
        cout << "\n(MAXV)\tNumber max of vehicles to use: " << MAXV;
        cout << "\n(MINV)\tNumber min of vehicles to use: " << MINV;
        cout << "\n\n--- Setting up the model ---\n";

        // Decision variables
        // Define 3D array binary variable X[i][j][k] for each node pair (i, j) and each vehicle k
        IloArray<IloArray<IloNumVarArray>> X(env, n);
        for (int i = 0; i < n; i++) {
            X[i] = IloArray<IloNumVarArray>(env, n);
            for (int j = 0; j < n; j++) {
                X[i][j] = IloNumVarArray(env, m, 0, 1, ILOBOOL);
                for (int k = 0; k < m; k++) {
                    char varName[30];  sprintf(varName, "X_%d_%d_%d", i, j, k);
                    X[i][j][k].setName(varName);
                    model.add(X[i][j][k]);
                }
            }
        }
        // Define 2D array continuous variable u[i][k] for each node i and each vehicle k
        IloArray<IloNumVarArray> u(env, n);
        for (int i = 1; i < n; i++) {
            u[i] = IloNumVarArray(env, m);
            for (int k = 0; k < m; k++) {
                std::string varName = "u(" + std::to_string(i) + "," + std::to_string(k) + ")";
                u[i][k] = IloNumVar(env, q[i], Q[k], varName.c_str());
                model.add(u[i][k]);
            }
        }

        // Objective function: minimize transportation costs
        IloExpr objective(env);
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) if (i != j) for (int k = 0; k < m; k++)
            objective += c[i][j] * X[i][j][k];
        model.add(IloMinimize(env, objective)); objective.end();

        // Constraints
        // 1. Each vehicle leaves the depot at most once
        for (int k = 0; k < m; k++) {
            IloExpr expression1(env);
            for (int i = 1; i < n; i++) expression1 += X[0][i][k];
            model.add(expression1 <= 1); expression1.end();
        }

        // 2. Flow conservation for each customer and vehicle
        for (int k = 0; k < m; k++) for (int j = 0; j < n; j++) {
                IloExpr expression2(env);
                for (int i = 0; i < n; i++) if (i != j) expression2 += X[i][j][k] - X[j][i][k];
                model.add(expression2 == 0); expression2.end();
        }

        // 3. Each customer is visited exactly once by any vehicle
        for (int j = 1; j < n; j++) {
            IloExpr expression3(env);
            for (int k = 0; k < m; k++) for (int i = 0; i < n; i++) if (i != j)
                 expression3 += X[i][j][k];
            model.add(expression3 == 1); expression3.end();
        }

        // 4. Capacity constraints for each vehicle
        for (int k = 0; k < m; k++) {
            IloExpr expression4(env);
            for (int j = 1; j < n; j++) for (int i = 0; i < n; i++) if (i != j)
                expression4 += X[i][j][k] * q[j];
            model.add(expression4 <= Q[k]); expression4.end();
        }

        // 5. Subtour elimination (MTZ constraints)
        for (int k = 0; k < m; k++) {
            for (int i = 1; i < n; i++) for (int j = 1; j < n; j++) if (i != j)
                 model.add(u[i][k] - u[j][k] + Q[k] * X[i][j][k] <= Q[k] - q[j]);
            for (int j = 1; j < n; j++)
                model.add(u[j][k] + (Q[k] - q[j]) * X[0][j][k] <= Q[k]);
        }

        // 6. Vehicle usage limits
        IloExpr expression6 (env);
        for (int k = 0; k < m; k++) for (int i = 1; i < n; i++) 
                expression6 += X[0][i][k];
        model.add(expression6 <= MAXV); model.add(expression6 >= MINV); expression6.end();

        // Solve the model
        IloCplex cplex(model);
        cplex.setParam(IloCplex::TiLim, 3600);  // Solution time limit (seconds)
        IloBool feasible = cplex.solve();
        if (feasible) {
            cout << "\nObjective (Total distance): " << cplex.getObjValue() << endl;
            for (int k = 0; k < m; k++) {
                cout << "\n\nRoute for vehicle " << k << ":\tQ[" << k << "] = " << Q[k];
                for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) if (i != j)
                    if (cplex.getValue(X[i][j][k]) > 0.5)
                        if (j > 0) cout << "\n\tFrom " << i << " to " << j << "\tq[" << j << "] = " << q[j] << "\tu[" << j << "," << k << "] = " << cplex.getValue(u[j][k]);
                else cout << "\n\tFrom " << i << " to " << j ;
            }
        }
        else
            cout << "No solution found within the time limit." << endl;
    }
    catch (IloException& e) {
        std::cerr << "Error: " << e << std::endl;
    }
    env.end();
}
