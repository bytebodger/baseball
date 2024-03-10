export const postgresConnection = {
    database: String(process.env.DB_NAME),
    password: String(process.env.DB_PASSWORD),
    user: String(process.env.DB_USER),
};
